import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau


def rank_correlation(scores_a, scores_b, method="spearman"):
    scores_a = np.asarray(scores_a)
    scores_b = np.asarray(scores_b)

    n = min(len(scores_a), len(scores_b))
    scores_a = scores_a[:n]
    scores_b = scores_b[:n]

    if method == "spearman":
        corr, p = spearmanr(scores_a, scores_b)
    elif method == "kendall":
        corr, p = kendalltau(scores_a, scores_b)
    else:
        raise ValueError("method must be 'spearman' or 'kendall'")

    return corr, p


def compare_explanations(df_a, df_b, key_cols=("id", "token_position")):
    merged = df_a.merge(
        df_b,
        on=list(key_cols),
        suffixes=("_a", "_b")
    )

    results = []

    for sample_id, group in merged.groupby("id"):
        corr, p = rank_correlation(
            group["attribution_a"].values,
            group["attribution_b"].values
        )

        results.append({
            "id": sample_id,
            "spearman": corr,
            "p_value": p,
            "n_tokens": len(group)
        })

    return pd.DataFrame(results)