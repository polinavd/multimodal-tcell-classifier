"""
Training script for Multimodal T-Cell Classifier
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pickle
import time
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

from models import MultimodalTCellClassifier, MultiTaskTCellModel, AdvancedMultimodalClassifier
from data import TCellDataset, TCellMultiTaskDataset, load_data, prepare_splits
from utils import compute_metrics, plot_confusion_matrix, plot_training_history


def train_epoch(model, loader, criterion, optimizer, device, multi_task=False, alpha=0.5):
    model.train()
    total_loss = 0
    correct, total = 0, 0
    
    for batch in loader:
        if multi_task:
            gex, tcr_a, tcr_b, labels, activation = batch
            activation = activation.to(device)
        else:
            gex, tcr_a, tcr_b, labels = batch
        
        gex = gex.to(device)
        tcr_a = tcr_a.to(device)
        tcr_b = tcr_b.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if multi_task:
            class_out, activation_out, _ = model(gex, tcr_a, tcr_b)
            cls_loss = criterion['cls'](class_out, labels)
            reg_loss = criterion['reg'](activation_out, activation)
            loss = alpha * cls_loss + (1 - alpha) * reg_loss
        else:
            outputs = model(gex, tcr_a, tcr_b)
            loss = criterion(outputs, labels)
            class_out = outputs
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = class_out.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device, multi_task=False):
    model.eval()
    total_loss = 0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            if multi_task:
                gex, tcr_a, tcr_b, labels, activation = batch
                activation = activation.to(device)
            else:
                gex, tcr_a, tcr_b, labels = batch
            
            gex = gex.to(device)
            tcr_a = tcr_a.to(device)
            tcr_b = tcr_b.to(device)
            labels = labels.to(device)
            
            if multi_task:
                class_out, activation_out, _ = model(gex, tcr_a, tcr_b)
                cls_loss = criterion['cls'](class_out, labels)
                reg_loss = criterion['reg'](activation_out, activation)
                loss = 0.5 * cls_loss + 0.5 * reg_loss
            else:
                outputs = model(gex, tcr_a, tcr_b)
                loss = criterion(outputs, labels)
                class_out = outputs
            
            total_loss += loss.item()
            _, predicted = class_out.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / total, np.array(all_preds), np.array(all_labels)


def main(config):
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\nLoading data...")
    data, label_encoder = load_data(config['data_dir'])
    splits = prepare_splits(data)
    
    # Create datasets
    multi_task = config.get('multi_task', False)
    
    if multi_task:
        train_dataset = TCellMultiTaskDataset(
            splits['train']['gex'], splits['train']['tcr_a'], 
            splits['train']['tcr_b'], splits['train']['y'],
            splits['train'].get('activation', np.zeros(len(splits['train']['y'])))
        )
        val_dataset = TCellMultiTaskDataset(
            splits['val']['gex'], splits['val']['tcr_a'],
            splits['val']['tcr_b'], splits['val']['y'],
            splits['val'].get('activation', np.zeros(len(splits['val']['y'])))
        )
        test_dataset = TCellMultiTaskDataset(
            splits['test']['gex'], splits['test']['tcr_a'],
            splits['test']['tcr_b'], splits['test']['y'],
            splits['test'].get('activation', np.zeros(len(splits['test']['y'])))
        )
    else:
        train_dataset = TCellDataset(
            splits['train']['gex'], splits['train']['tcr_a'],
            splits['train']['tcr_b'], splits['train']['y']
        )
        val_dataset = TCellDataset(
            splits['val']['gex'], splits['val']['tcr_a'],
            splits['val']['tcr_b'], splits['val']['y']
        )
        test_dataset = TCellDataset(
            splits['test']['gex'], splits['test']['tcr_a'],
            splits['test']['tcr_b'], splits['test']['y']
        )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                              shuffle=True, num_workers=config.get('num_workers', 0))
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                            shuffle=False, num_workers=config.get('num_workers', 0))
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                             shuffle=False, num_workers=config.get('num_workers', 0))
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Model
    print(f"\nInitializing {config['model']} model...")
    
    model_classes = {
        'basic': MultimodalTCellClassifier,
        'multitask': MultiTaskTCellModel,
        'advanced': AdvancedMultimodalClassifier
    }
    
    model = model_classes[config['model']](**config.get('model_params', {})).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    class_weights = compute_class_weight('balanced', 
                                         classes=np.unique(splits['train']['y']),
                                         y=splits['train']['y'])
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    if multi_task:
        criterion = {
            'cls': nn.CrossEntropyLoss(weight=class_weights),
            'reg': nn.MSELoss()
        }
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=config.get('scheduler_patience', 3), factor=0.5
    )
    
    # Training
    print("\nStarting training...")
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        start = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, multi_task
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, multi_task)
        
        scheduler.step(val_loss)
        elapsed = time.time() - start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), config['save_path'])
        else:
            patience_counter += 1
        
        if patience_counter >= config.get('early_stop_patience', 10):
            print("Early stopping")
            break
    
    # Test
    print("\n" + "="*50)
    print("TESTING")
    print("="*50)
    
    model.load_state_dict(torch.load(config['save_path']))
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, multi_task
    )
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=label_encoder.classes_))
    
    # Save results
    results_dir = Path(config.get('results_dir', 'results'))
    results_dir.mkdir(exist_ok=True)
    
    plot_confusion_matrix(test_labels, test_preds, label_encoder.classes_,
                          save_path=results_dir / 'confusion_matrix.png')
    plot_training_history(history, save_path=results_dir / 'training_history.png')
    
    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.device:
        config['device'] = args.device
    
    main(config)
