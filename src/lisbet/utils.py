"""LISBET"""

import numpy as np
from sklearn.metrics import f1_score
from scipy.optimize import linear_sum_assignment


def permute_predictions(predictions, labels):
    """ """
    # Permute HMM ids to match human labels
    n_states = np.max(predictions) + 1
    n_classes = np.max(labels) + 1

    hist2d = []
    for s in range(n_states):
        bin_pred = np.array(predictions == s, dtype=int)
        score = []
        for l in range(n_classes):
            bin_lab = np.array(labels == l, dtype=int)
            score.append(f1_score(bin_lab, bin_pred))
        hist2d.append(score)
    hist2d = np.array(hist2d)

    # Sort motifs to best match labels
    behavior_ids, motif_ids = linear_sum_assignment(hist2d.T, maximize=True)

    # Complete the assignment by adding the missing motifs
    missing_ids = list(set(np.unique(predictions)) - set(motif_ids))
    matched_ids = np.concatenate((motif_ids, missing_ids))

    # Match labels
    predictions_matched = np.array(predictions)
    for bhv, mtf in enumerate(matched_ids):
        predictions_matched[np.where(predictions == mtf)] = bhv

    return predictions_matched
