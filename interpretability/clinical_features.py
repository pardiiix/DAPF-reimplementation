import re
import pandas as pd
import numpy as np
import nltk
from collections import Counter


DISFLUENCY_MARKERS = {
    "uh", "um", "erm", "ah", "eh", "hmm",
    "xxx", "www", "unk"
}

EMPTY_SPEECH_MARKERS = {
    "thing", "something", "anything", "stuff", "someone",
    "somebody", "it", "that"
}


def type_token_ratio(tokens):
    clean = [t.lower() for t in tokens if t.isalpha()]
    if len(clean) == 0:
        return 0.0
    return len(set(clean)) / len(clean)


def mattr(tokens, window=50):
    clean = [t.lower() for t in tokens if t.isalpha()]
    if len(clean) < window:
        return type_token_ratio(clean)

    scores = []
    for i in range(len(clean) - window + 1):
        scores.append(len(set(clean[i:i + window])) / window)

    return float(np.mean(scores))


def feature_tags_for_tokens(tokens):
    rows = []

    pos_tags = nltk.pos_tag(tokens)

    for token, pos in pos_tags:
        tok = token.lower()

        rows.append({
            "token": token,
            "pos": pos,
            "is_pronoun": int(pos in {"PRP", "PRP$"}),
            "is_verb": int(pos.startswith("VB")),
            "is_noun": int(pos.startswith("NN")),
            "is_disfluency": int(tok in DISFLUENCY_MARKERS),
            "is_empty_speech": int(tok in EMPTY_SPEECH_MARKERS),
            "is_repetition_marker": int(tok in {"again", "repeat", "same"})
        })

    return pd.DataFrame(rows)


def attribution_feature_enrichment(attr_df, top_percent=0.10):
    """
    attr_df requires:
    id, token, attribution
    """

    rows = []

    for sample_id, group in attr_df.groupby("id"):
        group = group.copy()
        group["abs_attr"] = group["attribution"].abs()

        threshold = group["abs_attr"].quantile(1 - top_percent)
        top = group[group["abs_attr"] >= threshold]

        feature_df = feature_tags_for_tokens(group["token"].tolist())
        top_feature_df = feature_tags_for_tokens(top["token"].tolist())

        rows.append({
            "id": sample_id,
            "top_percent": top_percent,
            "top_pronoun_rate": top_feature_df["is_pronoun"].mean(),
            "all_pronoun_rate": feature_df["is_pronoun"].mean(),
            "top_disfluency_rate": top_feature_df["is_disfluency"].mean(),
            "all_disfluency_rate": feature_df["is_disfluency"].mean(),
            "top_empty_speech_rate": top_feature_df["is_empty_speech"].mean(),
            "all_empty_speech_rate": feature_df["is_empty_speech"].mean()
        })

    return pd.DataFrame(rows)