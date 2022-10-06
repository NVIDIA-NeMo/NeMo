# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score


def auc_roc(y_true, y_score):
    """Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
    """
    return roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):
    """Compute Area Under the Precision-Recall Curve (PR AUC) from prediction scores.
    """
    return average_precision_score(y_true, y_score)


def auc_nt(y_true, y_score):
    """Compute Area Under the Negative Predictive Value vs. True Negative Rate Curve (NT AUC) from prediction scores.

    This metric can be thought of as a PR AUC in which errors are treated as positives.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    return average_precision_score(1 - y_true, 1 - y_score)


def nce(y_true, y_score):
    """Compute Normalized Cross Entropy (NCE) from prediction scores. Also known as the Normalized Mutual Information.

    NCE measures how close the correct prediction scores are to one and the incorrect prediction scores are to zero.
    Negative NCE values indicate that the classifier performs worse than the setting all prediction scores
    as the proportion of correct predictions.
    """
    p = sum(y_true) / len(y_true)
    eps = 1e-15
    Hp = -(math.log(p + eps) * p + math.log(1 - p + eps) * (1 - p))
    return (Hp - log_loss(y_true, y_score)) / Hp


def ece(y_true, y_score, n_bins=100):
    """Compute Expected Calibration Error (ECE) from prediction scores.

    ECE measures how close the correct prediction scores are to one and the incorrect prediction scores are to zero.
    ECE ranges from zero to one with the best value zero (the lower the value, the better).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    py = np.array([1 - y_score, y_score]).T
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        threshold = (a + b) / 2
        py_index = (py.T[1] >= threshold).astype(int)
        py_value = py[np.arange(len(py_index)), py_index]
        bin_range = ((py_value > a) & (py_value <= b)).nonzero()[0]
        Bm[m] = len(bin_range)
        if Bm[m] > 0:
            acc[m] = (py_index[bin_range] == y_true[bin_range]).sum()
            conf[m] = py_value[bin_range].sum()
        if Bm[m] != 0:
            acc[m] /= Bm[m]
            conf[m] /= Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


def auc_yc(y_true, y_score, return_std_maximum=False, return_curve=False, n_bins=100):
    """Compute Area Under the Youden's Curve (YC AUC) from prediction scores.

    YC AUC represents the rate of the effective threshold range.

    If return_std_maximum is set to True, std and maximum values of the Youden's Curve are returned with the AUC.
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    mask_correct = y_true == 1
    count_correct = len(mask_correct.nonzero()[0])
    count_incorrect = len(y_true) - count_correct
    y_score_correct = y_score[mask_correct]
    y_score_incorrect = y_score[~mask_correct]
    yc = []
    thresholds = [i / n_bins for i in range(0, n_bins + 1)]
    for threshold in thresholds:
        tnr = len((np.array(y_score_incorrect) < threshold).nonzero()[0]) / count_incorrect
        fnr = len((np.array(y_score_correct) < threshold).nonzero()[0]) / count_correct
        yc.append(tnr - fnr)
    yc = np.array(yc)
    if return_std_maximum and return_curve:
        return yc.mean(), yc.max(), yc.std(), (thresholds, yc)
    elif return_std_maximum:
        return yc.mean(), yc.max(), yc.std()
    elif return_curve:
        return yc.mean(), (thresholds, yc)
    else:
        return yc.mean()
