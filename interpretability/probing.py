import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def run_linear_probe(X, y, scoring="f1_macro", n_splits=5):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        clf,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": scores.tolist()
    }

def run_random_label_probe(
    features,
    labels,
    n_repeats=100,
    random_seed=42,
):
    """
    Random-label sanity control for representation probing.

    Uses the same probe configuration as the main probing experiment:
      - StandardScaler
      - balanced logistic regression
      - five-fold stratified cross-validation

    The original diagnosis labels are randomly permuted, breaking the
    relationship between the representations and diagnosis.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import StratifiedKFold

    features = np.asarray(features)
    labels = np.asarray(labels)

    rng = np.random.default_rng(random_seed)

    # Create the folds once using the real labels.
    # The same participant folds are reused for every permutation.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )
    fixed_splits = list(cv.split(features, labels))

    repeat_scores = []

    for _ in range(n_repeats):
        shuffled_labels = rng.permutation(labels)

        fold_scores = []

        for train_indices, test_indices in fixed_splits:
            y_train = shuffled_labels[train_indices]
            y_test = shuffled_labels[test_indices]

            # Extremely unlikely, but avoid fitting if a shuffled
            # training fold accidentally contains only one class.
            if len(np.unique(y_train)) < 2:
                continue

            probe = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            )

            probe.fit(
                features[train_indices],
                y_train,
            )

            predictions = probe.predict(
                features[test_indices]
            )

            fold_scores.append(
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

        if len(fold_scores) == 5:
            repeat_scores.append(
                float(np.mean(fold_scores))
            )

    if not repeat_scores:
        raise RuntimeError(
            "No valid random-label probing repetitions were completed."
        )

    repeat_scores = np.asarray(
        repeat_scores,
        dtype=float,
    )

    return {
        "mean_macro_f1": float(repeat_scores.mean()),
        "std_macro_f1": float(repeat_scores.std(ddof=1)),
        "ci_95_low": float(
            np.quantile(repeat_scores, 0.025)
        ),
        "ci_95_high": float(
            np.quantile(repeat_scores, 0.975)
        ),
        "n_valid_repeats": int(len(repeat_scores)),
        "all_scores": repeat_scores.tolist(),
    }