import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error for binary classification.

    Args:
        y_true:
            Ground-truth labels, expected as 0/1.
        y_prob:
            Predicted probability for the positive class, e.g. P(dementia).
        n_bins:
            Number of confidence bins.

    Returns:
        Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    confidences = np.maximum(y_prob, 1 - y_prob)
    predictions = (y_prob >= 0.5).astype(int)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)

        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += mask.mean() * abs(bin_acc - bin_conf)

    return float(ece)
