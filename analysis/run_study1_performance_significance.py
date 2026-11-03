import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


DAPF_RUN = (
    "bert-base-uncased_tempmanual7_verbmanual_epoch10_optimadamw_"
    "stk16_100_domain2_bs4_prlr0.5_joint_cvFalse"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dapf_root", default="output_multiseed")
    p.add_argument("--bert_cls_root", default="output_bert_cls_multiseed")
    p.add_argument(
        "--bert_cls_prompt_root",
        default="output_bert_cls_prompt_multiseed"
    )
    p.add_argument("--output_dir", default="performance_significance")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--n_permutations", type=int, default=100000)
    p.add_argument("--random_seed", type=int, default=2026)
    return p.parse_args()


def dapf_path(root, seed):
    return (
        Path(root)
        / f"seed_{seed}"
        / DAPF_RUN
        / f"version_{seed}"
        / "checkpoints"
        / "epoch-1"
        / "test_results.csv"
    )


def baseline_path(root, seed):
    return Path(root) / f"seed_{seed}" / "test_results.csv"


def load_file(path):
    df = pd.read_csv(path)

    required = {"id", "labels", "pred_labels"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    if len(df) != 48:
        raise ValueError(
            f"{path}: expected 48 test examples, got {len(df)}"
        )

    return df


def extract_probabilities(df):
    if "probability_dementia" in df.columns:
        return df["probability_dementia"].astype(float).to_numpy()

    out = []

    for value in df["probas"]:
        if isinstance(value, str):
            try:
                value = ast.literal_eval(value)
            except Exception:
                pass

        a = np.asarray(value).reshape(-1)

        if len(a) == 1:
            out.append(float(a[0]))
        elif len(a) == 2:
            out.append(float(a[1]))
        else:
            raise ValueError(
                f"Unexpected probability representation: {value}"
            )

    return np.asarray(out)


def metrics(df):
    y = df["labels"].astype(int).to_numpy()
    pred = df["pred_labels"].astype(int).to_numpy()
    prob = extract_probabilities(df)

    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "auroc": roc_auc_score(y, prob),
    }


def assert_aligned(a, b, name_a, name_b):
    if not np.array_equal(
        a["id"].astype(str).to_numpy(),
        b["id"].astype(str).to_numpy()
    ):
        raise ValueError(f"IDs do not align: {name_a} vs {name_b}")

    if not np.array_equal(
        a["labels"].astype(int).to_numpy(),
        b["labels"].astype(int).to_numpy()
    ):
        raise ValueError(f"Labels do not align: {name_a} vs {name_b}")


def paired_permutation_metric(
    df_a,
    df_b,
    metric_name,
    rng,
    n_permutations=100000,
):
    """
    Paired randomization test.

    Under H0, for each test participant, predictions from model A and
    model B are exchangeable. We randomly swap the two models'
    predictions participant-by-participant.

    Two-sided p-value:
        proportion of permutations whose absolute metric difference
        is at least as large as the observed absolute difference.
    """

    y = df_a["labels"].astype(int).to_numpy()

    pred_a = df_a["pred_labels"].astype(int).to_numpy()
    pred_b = df_b["pred_labels"].astype(int).to_numpy()

    if metric_name == "accuracy":
        metric = accuracy_score
    elif metric_name == "macro_f1":
        metric = lambda yy, pp: f1_score(
            yy, pp, average="macro"
        )
    else:
        raise ValueError(metric_name)

    observed = metric(y, pred_a) - metric(y, pred_b)

    extreme = 0

    for _ in range(n_permutations):
        swap = rng.integers(0, 2, size=len(y)).astype(bool)

        perm_a = pred_a.copy()
        perm_b = pred_b.copy()

        perm_a[swap] = pred_b[swap]
        perm_b[swap] = pred_a[swap]

        diff = metric(y, perm_a) - metric(y, perm_b)

        if abs(diff) >= abs(observed) - 1e-12:
            extreme += 1

    p = (extreme + 1) / (n_permutations + 1)

    return observed, p


def exact_mcnemar(df_a, df_b):
    """
    Exact two-sided McNemar test based on discordant correctness.
    """

    y = df_a["labels"].astype(int).to_numpy()
    pa = df_a["pred_labels"].astype(int).to_numpy()
    pb = df_b["pred_labels"].astype(int).to_numpy()

    correct_a = pa == y
    correct_b = pb == y

    # A correct, B wrong
    b = int(np.sum(correct_a & ~correct_b))

    # A wrong, B correct
    c = int(np.sum(~correct_a & correct_b))

    n = b + c

    if n == 0:
        p = 1.0
    else:
        p = binomtest(
            min(b, c),
            n=n,
            p=0.5,
            alternative="two-sided"
        ).pvalue

    return b, c, p


def holm_adjust(pvalues):
    """
    Holm family-wise error correction.
    """

    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)

    order = np.argsort(pvalues)
    adjusted = np.empty(m)

    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in args.seeds.split(",")]

    model_data = {
        "DAPF": {},
        "BERT-CLS": {},
        "BERT-CLS+Prompt": {},
    }

    metric_rows = []

    # ---------------------------------------------------------
    # Load data and verify alignment
    # ---------------------------------------------------------

    for seed in seeds:
        d = load_file(dapf_path(args.dapf_root, seed))
        c = load_file(baseline_path(args.bert_cls_root, seed))
        p = load_file(
            baseline_path(args.bert_cls_prompt_root, seed)
        )

        assert_aligned(d, c, "DAPF", "BERT-CLS")
        assert_aligned(d, p, "DAPF", "BERT-CLS+Prompt")

        model_data["DAPF"][seed] = d
        model_data["BERT-CLS"][seed] = c
        model_data["BERT-CLS+Prompt"][seed] = p

        for name, df in [
            ("DAPF", d),
            ("BERT-CLS", c),
            ("BERT-CLS+Prompt", p),
        ]:
            m = metrics(df)

            metric_rows.append({
                "model": name,
                "seed": seed,
                **m,
            })

    metric_df = pd.DataFrame(metric_rows)

    metric_df.to_csv(
        output_dir / "per_seed_metrics.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Sanity check against Table 3
    # ---------------------------------------------------------

    means = (
        metric_df
        .groupby("model")[["accuracy", "macro_f1", "auroc"]]
        .mean()
    )

    expected = {
        "DAPF": {
            "accuracy": 0.8292,
            "macro_f1": 0.8287,
            "auroc": 0.8757,
        },
        "BERT-CLS": {
            "accuracy": 0.8083,
            "macro_f1": 0.8063,
            "auroc": 0.8906,
        },
        "BERT-CLS+Prompt": {
            "accuracy": 0.8083,
            "macro_f1": 0.8065,
            "auroc": 0.8819,
        },
    }

    for model, target in expected.items():
        for metric_name, target_value in target.items():

            actual = means.loc[model, metric_name]

            if not np.isclose(
                actual,
                target_value,
                atol=0.00011,
            ):
                raise RuntimeError(
                    f"TABLE 3 SANITY CHECK FAILED: "
                    f"{model} {metric_name}: "
                    f"computed={actual:.6f}, "
                    f"expected≈{target_value:.4f}"
                )

    print("\nTable 3 sanity check: PASSED")
    print(means)

    # ---------------------------------------------------------
    # Participant-level paired tests, separately within seed
    # ---------------------------------------------------------

    rng = np.random.default_rng(args.random_seed)

    comparisons = [
        ("DAPF", "BERT-CLS"),
        ("DAPF", "BERT-CLS+Prompt"),
    ]

    rows = []

    for model_a, model_b in comparisons:

        for seed in seeds:

            a = model_data[model_a][seed]
            bdf = model_data[model_b][seed]

            ma = metrics(a)
            mb = metrics(bdf)

            # Macro-F1 paired randomization
            diff_f1, p_f1 = paired_permutation_metric(
                a,
                bdf,
                "macro_f1",
                rng,
                args.n_permutations,
            )

            rows.append({
                "comparison": f"{model_a} vs {model_b}",
                "seed": seed,
                "test": "paired_permutation",
                "metric": "macro_f1",
                "model_a_value": ma["macro_f1"],
                "model_b_value": mb["macro_f1"],
                "difference": diff_f1,
                "p_value": p_f1,
            })

            # Accuracy paired randomization
            diff_acc, p_acc = paired_permutation_metric(
                a,
                bdf,
                "accuracy",
                rng,
                args.n_permutations,
            )

            rows.append({
                "comparison": f"{model_a} vs {model_b}",
                "seed": seed,
                "test": "paired_permutation",
                "metric": "accuracy",
                "model_a_value": ma["accuracy"],
                "model_b_value": mb["accuracy"],
                "difference": diff_acc,
                "p_value": p_acc,
            })

            # Exact McNemar
            discord_a, discord_b, p_mc = exact_mcnemar(
                a,
                bdf
            )

            rows.append({
                "comparison": f"{model_a} vs {model_b}",
                "seed": seed,
                "test": "exact_mcnemar",
                "metric": "accuracy",
                "model_a_value": ma["accuracy"],
                "model_b_value": mb["accuracy"],
                "difference": (
                    ma["accuracy"] - mb["accuracy"]
                ),
                "discordant_a_correct": discord_a,
                "discordant_b_correct": discord_b,
                "p_value": p_mc,
            })

    tests = pd.DataFrame(rows)

    # Holm correction within test/metric family
    tests["p_holm"] = np.nan

    for (test_name, metric_name), idx in tests.groupby(
        ["test", "metric"]
    ).groups.items():

        idx = list(idx)

        tests.loc[idx, "p_holm"] = holm_adjust(
            tests.loc[idx, "p_value"].to_numpy()
        )

    tests.to_csv(
        output_dir / "participant_level_tests.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Seed-level descriptive differences
    # ---------------------------------------------------------

    seed_rows = []

    for model_a, model_b in comparisons:

        a = (
            metric_df[metric_df.model == model_a]
            .set_index("seed")
        )

        bdf = (
            metric_df[metric_df.model == model_b]
            .set_index("seed")
        )

        for metric_name in [
            "accuracy",
            "macro_f1",
            "auroc",
        ]:
            diffs = (
                a.loc[seeds, metric_name].to_numpy()
                - bdf.loc[seeds, metric_name].to_numpy()
            )

            seed_rows.append({
                "comparison": f"{model_a} vs {model_b}",
                "metric": metric_name,
                "mean_difference": np.mean(diffs),
                "std_difference": np.std(diffs, ddof=1),
                "n_seeds": len(diffs),
                "differences": repr(diffs.tolist()),
            })

    seed_df = pd.DataFrame(seed_rows)

    seed_df.to_csv(
        output_dir / "seed_level_differences.csv",
        index=False
    )

    # ---------------------------------------------------------
    # Console output
    # ---------------------------------------------------------

    print("\nParticipant-level tests:")
    print(
        tests[
            [
                "comparison",
                "seed",
                "test",
                "metric",
                "difference",
                "p_value",
                "p_holm",
            ]
        ].to_string(index=False)
    )

    print("\nSeed-level differences:")
    print(seed_df.to_string(index=False))

    print(
        f"\nSaved all results to: {output_dir}"
    )


if __name__ == "__main__":
    main()