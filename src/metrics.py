import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

class TriggerMetrics:
    """
    Computes performance metrics for any trigger system based on ground truth vs predictions.
    """
    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Args:
            y_true: True binary labels (1=signal, 0=background)
            y_pred: Predicted binary labels (1=accept, 0=reject)
            
        Returns:
            dict containing multiple performance metrics.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # 'Efficiency' is generally physics jargon for recall / true positive rate
        efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Background Rejection Rate (True Negative Rate)
        bg_rejection = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            'True Positives': tp,
            'False Positives': fp,
            'True Negatives': tn,
            'False Negatives': fn,
            'Precision': precision,
            'Recall': recall,
            'Efficiency': efficiency,
            'F1 Score': f1,
            'Background Rejection': bg_rejection
        }
