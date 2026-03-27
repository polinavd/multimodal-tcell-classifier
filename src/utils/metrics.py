"""
Evaluation metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, r2_score
)
from scipy.stats import pearsonr


def compute_metrics(y_true, y_pred, y_prob=None):
    """Compute classification metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'recall_macro': recall_score(y_true, y_pred, average='macro')
    }
    return metrics


def classification_report_dict(y_true, y_pred, target_names=None):
    """Get classification report as dictionary"""
    return classification_report(y_true, y_pred, target_names=target_names, output_dict=True)


def regression_metrics(y_true, y_pred):
    """Compute regression metrics"""
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    pearson_r, p_value = pearsonr(y_true, y_pred)
    
    return {
        'r2': r2,
        'mae': mae,
        'pearson_r': pearson_r,
        'p_value': p_value
    }
