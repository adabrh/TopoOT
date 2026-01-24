from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def compute_metrics_binary(gt_bin, pred_bin):
    gt_bin = np.asarray(gt_bin).astype(int).ravel()
    pred_bin = np.asarray(pred_bin).astype(int).ravel()
    prec = precision_score(gt_bin, pred_bin, zero_division=0)
    rec  = recall_score(gt_bin, pred_bin, zero_division=0)
    f1   = f1_score(gt_bin, pred_bin, zero_division=0)
    return {'precision': float(prec), 'recall': float(rec), 'f1': float(f1)}
