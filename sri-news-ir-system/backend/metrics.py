import numpy as np
from sklearn.metrics import confusion_matrix


def precision_at_k(scores, relevance, k=5):
    idx = np.argsort(scores)[::-1][:k]
    return relevance[idx].sum() / k


def recall_at_k(scores, relevance, k=5):
    idx = np.argsort(scores)[::-1][:k]
    return relevance[idx].sum() / relevance.sum()


def average_precision(scores, relevance):
    sorted_idx = np.argsort(scores)[::-1]
    rel = relevance[sorted_idx]

    precisions = []
    hits = 0
    for i, r in enumerate(rel):
        if r == 1:
            hits += 1
            precisions.append(hits / (i + 1))

    return np.mean(precisions) if precisions else 0


def sri_confusion_matrix(scores, relevance, threshold=None):
    if threshold is None:
        threshold = np.percentile(scores, 50)
    preds = (scores >= threshold).astype(int)
    return confusion_matrix(relevance, preds)
