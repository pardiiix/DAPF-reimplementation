import re
import numpy as np
import pandas as pd
import torch


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def mask_token_span(tokens, start, end, mask_token="[MASK]"):
    new_tokens = tokens.copy()
    new_tokens[start:end] = [mask_token]
    return new_tokens


def probability_of_dementia(prompt_model, batch, device):
    prompt_model.eval()

    with torch.no_grad():
        batch = batch.to(device)
        logits = prompt_model(batch)
        probs = torch.softmax(logits, dim=-1)

    return probs[:, 1].detach().cpu().numpy()


def occlusion_scores_for_tokens(
    original_prob,
    tokenizer,
    prompt_model,
    wrapped_input_builder,
    text,
    window_size=5,
    stride=5,
    device="cuda"
):
    """
    wrapped_input_builder should rebuild the OpenPrompt input for the perturbed text.
    """

    tokens = tokenizer.tokenize(text)
    rows = []

    for start in range(0, len(tokens), stride):
        end = min(start + window_size, len(tokens))
        perturbed_tokens = tokens.copy()
        perturbed_tokens[start:end] = [tokenizer.mask_token] * (end - start)

        perturbed_text = tokenizer.convert_tokens_to_string(perturbed_tokens)

        batch = wrapped_input_builder(perturbed_text)
        perturbed_prob = probability_of_dementia(prompt_model, batch, device)[0]

        rows.append({
            "start": start,
            "end": end,
            "occluded_text": tokenizer.convert_tokens_to_string(tokens[start:end]),
            "original_prob": original_prob,
            "perturbed_prob": perturbed_prob,
            "importance": original_prob - perturbed_prob
        })

    return pd.DataFrame(rows)


def leave_one_sentence_out(
    original_prob,
    prompt_model,
    wrapped_input_builder,
    text,
    device="cuda"
):
    sentences = split_sentences(text)
    rows = []

    for i, sent in enumerate(sentences):
        perturbed = " ".join([s for j, s in enumerate(sentences) if j != i])
        batch = wrapped_input_builder(perturbed)
        perturbed_prob = probability_of_dementia(prompt_model, batch, device)[0]

        rows.append({
            "sentence_index": i,
            "sentence": sent,
            "original_prob": original_prob,
            "perturbed_prob": perturbed_prob,
            "importance": original_prob - perturbed_prob
        })

    return pd.DataFrame(rows)

def deletion_curve(
    token_scores,
    tokens,
    predict_fn,
    steps=(0.05, 0.10, 0.20, 0.30, 0.50),
    use_abs=True
):
    """
    Remove top-attributed tokens and observe probability change.

    Args:
        token_scores: attribution score per token.
        tokens: token list.
        predict_fn: function that takes a token list and returns P(dementia).
        steps: fractions of tokens to remove.
        use_abs: if True, rank by absolute attribution magnitude.

    Returns:
        DataFrame with probability after deleting top-k tokens.
    """
    scores = np.array(token_scores)

    if use_abs:
        ranking_scores = np.abs(scores)
    else:
        ranking_scores = scores

    order = np.argsort(-ranking_scores)
    rows = []

    original_prob = predict_fn(tokens)

    for frac in steps:
        k = max(1, int(len(tokens) * frac))
        remove_idx = set(order[:k])

        perturbed_tokens = [
            tok for i, tok in enumerate(tokens)
            if i not in remove_idx
        ]

        prob = predict_fn(perturbed_tokens)

        rows.append({
            "fraction_removed": frac,
            "num_removed": k,
            "original_probability": original_prob,
            "perturbed_probability": prob,
            "probability_drop": original_prob - prob
        })

    return pd.DataFrame(rows)


def insertion_curve(
    token_scores,
    tokens,
    baseline_token,
    predict_fn,
    steps=(0.05, 0.10, 0.20, 0.30, 0.50),
    use_abs=True
):
    """
    Start from a baseline sequence and insert top-attributed tokens.

    Args:
        token_scores: attribution score per token.
        tokens: token list.
        baseline_token: token used for non-inserted positions, e.g. tokenizer.mask_token.
        predict_fn: function that takes a token list and returns P(dementia).
        steps: fractions of tokens to insert.
        use_abs: if True, rank by absolute attribution magnitude.

    Returns:
        DataFrame with probability after inserting top-k tokens.
    """
    scores = np.array(token_scores)

    if use_abs:
        ranking_scores = np.abs(scores)
    else:
        ranking_scores = scores

    order = np.argsort(-ranking_scores)
    rows = []

    original_prob = predict_fn(tokens)
    baseline_prob = predict_fn([baseline_token for _ in tokens])

    for frac in steps:
        k = max(1, int(len(tokens) * frac))
        keep_idx = set(order[:k])

        perturbed_tokens = [
            tok if i in keep_idx else baseline_token
            for i, tok in enumerate(tokens)
        ]

        prob = predict_fn(perturbed_tokens)

        rows.append({
            "fraction_inserted": frac,
            "num_inserted": k,
            "original_probability": original_prob,
            "baseline_probability": baseline_prob,
            "perturbed_probability": prob,
            "probability_recovered": prob - baseline_prob
        })

    return pd.DataFrame(rows)