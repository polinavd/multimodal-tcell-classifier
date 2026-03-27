"""
Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=True, 
                          figsize=(10, 8), save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return plt.gcf()


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig


def plot_umap(embeddings, labels, title='UMAP', figsize=(10, 8), save_path=None):
    """Plot UMAP visualization"""
    import scanpy as sc
    import anndata as ad
    
    adata = ad.AnnData(embeddings)
    adata.obs['label'] = labels
    
    sc.pp.neighbors(adata, use_rep='X')
    sc.tl.umap(adata)
    
    plt.figure(figsize=figsize)
    sc.pl.umap(adata, color='label', title=title, show=False)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return plt.gcf()


def plot_ablation_study(results, save_path=None):
    """Plot ablation study results"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = list(results.keys())
    accuracies = [results[m] * 100 for m in models]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Ablation Study: Contribution of Each Modality', fontsize=14)
    ax.set_ylim(0, 100)
    ax.axhline(y=14.3, color='gray', linestyle='--', label='Random baseline')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    return fig
