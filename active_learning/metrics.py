"""Metric functions for training and evaluation
TODO these should either all work in torch or numpy (not both as will be confusing)
TODO use argpartition instead of argsort (-ve)
"""

import torch
import numpy as np

from scipy.stats import entropy

from sklearn.metrics import precision_recall_fscore_support, f1_score


def get_metrics(metric):
    """Get appropriate metric function
    """
    if metric == "entropy":
        return pred_entropy
    elif metric == "lc":
        return least_confidence
    elif metric == "margin":
        return margin
    else:
        raise NameError(f"metric {metric} not implemented")


def cls_metrics(y_true, y_pred, num_classes, average=None):
    """compute precision, recall and f1 score (sklearn)
    assumes labels are 0, 1, ..., num_classes - 1
    """
    return precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), average=average)


def f1(y_true, y_pred, average="macro"):
    """Compute F1 score (sklearn)
    """
    return f1_score(y_true, y_pred, average=average)


def accuracy(y_true, y_pred):
    """Compute classification accuracy (pytorch tensors)
    """
    return torch.mean(y_pred.eq(y_true).float())


def least_confidence(probs):
    """Compute least confidence scores 1-p(y*|x) of a set of predictions
    Params:
    - probs (np.array): array of model output probabilities (B, num_classes)
    Returns:
    - scores (np.array): array of scores (B,)
    """
    return 1 - np.max(probs, axis=-1)


def margin(probs):
    """Compute margin scores p(y1|x)-p(y2|x) of a set of predictions
    Params:
    - probs (np.array): array of model output probabilities (B, num_classes)
    Returns:
    - scores (np.array): array of scores given as -margin as margin should be minimised) (B,)
    """
    # sorted in decreasing order using -ve
    part = np.partition(-probs, 1, axis=-1)
    # undo the -ve to get margin
    margin = - part[:, 0] + part[:, 1]
    # return -ve margin as a score as lower margin is more uncertain
    return - margin


def pred_entropy(probs):
    """Compute entropy of a set of predictions
    Params:
    - probs (np.array): array of model output probabilities (B, num_classes)
    Returns:
    - entropies (np.array): array of entropies (B,)
    """
    # return entropy (natural log)
    return entropy(probs, axis=-1)


if __name__ == "__main__":
    pass
