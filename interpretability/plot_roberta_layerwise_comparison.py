
#!/usr/bin/env python3
"""
Compare layer-wise diagnosis probes for three RoBERTa architectures.

Models:
    DAPF-RoBERTa: CLS and supervised MASK
    RoBERTa-CLS: CLS only
    RoBERTa-CLS+Prompt: CLS and auxiliary MASK

Uses five independent seeds.
Error bars show sample SD across five seed-level probe means.

No training or inference is performed.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (0, 1, 2, 3, 4)
LAYERS = np.arange(1, 13)

DAPF = "DAPF-RoBERTa"
CLS = "RoBERTa-CLS"
PROMPT = "RoBERTa-CLS+Prompt"

MODELS = (DAPF, CLS, PROMPT)

REPS = {
    DAPF: ("CLS", "MASK"),
    CLS: ("CLS",),
    PROMPT: ("CLS", "MASK"),
}

REP_COLORS = {
    "CLS": "#236da6",
    "MASK": "#d47515",
}

MODEL_COLORS = {
    DAPF: "#236da6",
    CLS: "#2f8a53",
    PROMPT: "#b63739",
}


def read_probe(path, model):
    """Read and validate one layer-wise probe CSV."""

    df = pd.read_csv(path)

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    required = ("layer", "representation")

    missing = [
        name for name in required
        if name not in lookup
    ]

    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}"
        )

    # Support both existing CSV formats.
    if "score_mean" in lookup:
        score_col = lookup["score_mean"]

    elif "mean_score" in lookup:
        score_col = lookup["mean_score"]

    elif "macro_f1" in lookup:
        score_col = lookup["macro_f1"]

    else:
        raise ValueError(
            f"{path}: no mean probe macro-F1 column: "
            f"{list(df.columns)}"
        )

    if "target" in lookup:
        diagnostic_rows = (
            df[lookup["target"]]
            .astype(str)
            .str.contains(
                "diagnos|dementia",
                case=False,
                regex=True
            )
        )

        if diagnostic_rows.any():
            df = df.loc[diagnostic_rows].copy()

    out = df[[
        lookup["layer"],
        lookup["representation"],
        score_col
    ]].copy()

    out.columns = [
        "layer",
        "representation",
        "macro_f1"
    ]

    out["layer"] = pd.to_numeric(
        out["layer"], errors="raise"
    ).astype(int)

    out["macro_f1"] = pd.to_numeric(
        out["macro_f1"], errors="raise"
    )

    out["representation"] = (
        out["representation"]
        .astype(str)
        .str.lower()
        .map(
            lambda x:
                "MASK" if "mask" in x
                else "CLS" if "cls" in x
                else None
        )
    )

    # Support zero-based and one-based layer indices.
    if set(out["layer"]) == set(range(12)):
        out["layer"] += 1

    expected = {
        (i, r)
        for i in LAYERS
        for r in REPS[model]
    }

    actual = list(zip(
        out["layer"],
        out["representation"]
    ))

    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            f"{path}: expected {len(expected)} "
            f"layer/representation pairs for {model}, "
            f"found {len(actual)}: {set(actual)}"
        )

    if (
        not np.isfinite(out["macro_f1"]).all()
        or not out["macro_f1"].between(0, 1).all()
    ):
        raise ValueError(
            f"{path}: invalid macro-F1 values"
        )

    return out


def load_model(root, model):
    """Load five seeds of one model."""

    result = []

    for seed in SEEDS:
        folder = root / f"seed_{seed}"

        candidates = [
            folder / "layerwise_probe_results.csv",
            folder / "layerwise_probing"
            / "layerwise_probe_results.csv",
        ]

        file = next(
            (p for p in candidates if p.is_file()),
            None
        )

        if file is None:
            raise FileNotFoundError(
                f"No layerwise_probe_results.csv "
                f"found for {model}, seed {seed}, "
                f"under {folder}"
            )

        frame = read_probe(file, model)

        frame["seed"] = seed
        frame["model"] = model

        result.append(frame)

        print(
            f"Loaded {model}, seed {seed}: "
            f"{file} ({len(frame)} rows)"
        )

    return pd.concat(result, ignore_index=True)


def save(fig, out, stem):
    """Save each figure as PNG and PDF."""

    for extension in ("png", "pdf"):
        path = out / f"{stem}.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved {path}")

    plt.close(fig)


def get_curve(summary, model, rep):
    """Get mean and SD across five seeds."""

    sub = summary[
        (summary["model"] == model)
        & (summary["representation"] == rep)
    ].sort_values("layer")

    if (
        len(sub) != 12
        or not (sub["n_seeds"] == 5).all()
    ):
        raise ValueError(
            f"Incomplete five-seed curve: {model} / {rep}"
        )

    return (
        sub["layer"].to_numpy(),
        sub["mean"].to_numpy(),
        sub["sd"].to_numpy()
    )


def error_line(
    ax,
    x,
    y,
    sd,
    label,
    color,
    marker,
    offset=0.0,
    linestyle="-"
):
    """Draw mean curve with SD error bars."""

    ax.errorbar(
        x + offset,
        y,
        yerr=sd,
        label=label,
        color=color,
        marker=marker,
        linestyle=linestyle,
        markersize=4.5,
        linewidth=1.8,
        elinewidth=0.85,
        capsize=2.5,
        capthick=0.85,
    )


def format_layer_axes(ax):
    ax.set_xticks(LAYERS)
    ax.set_xlim(0.55, 12.45)
    ax.set_ylim(0.56, 0.94)
    ax.set_xlabel("Transformer layer (1–12)")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)


def draw_model_panel(ax, summary, model):
    """Draw representations belonging to one model."""

    for rep, offset, marker in (
        ("CLS", -0.06, "o"),
        ("MASK", 0.06, "s"),
    ):
        if rep not in REPS[model]:
            continue

        x, y, sd = get_curve(
            summary, model, rep
        )

        if rep == "CLS":
            label = "CLS"
        elif model == DAPF:
            label = "Supervised MASK"
        else:
            label = "Auxiliary MASK"

        error_line(
            ax,
            x,
            y,
            sd,
            label,
            REP_COLORS[rep],
            marker,
            offset
        )

    format_layer_axes(ax)

    ax.set_title(model, fontsize=11)

    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=8
    )


def plot_layers(summary, out):
    """Generate layer-wise comparison figures."""

    # Main three-panel figure.
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17.4, 4.7),
        sharey=True
    )

    for ax, model in zip(axes, MODELS):
        draw_model_panel(ax, summary, model)

    axes[0].set_ylabel(
        "Diagnosis probe macro-F1"
    )

    fig.suptitle(
        "RoBERTa representation probing across "
        "five seeds (mean ± SD)",
        fontsize=14
    )

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save(
        fig,
        out,
        "roberta_layerwise_cls_mask"
    )

    # Separate model figures.
    for model, stem in (
        (DAPF, "roberta_dapf_layerwise_cls_mask"),
        (CLS, "roberta_cls_layerwise_cls"),
        (PROMPT, "roberta_cls_prompt_layerwise_cls_mask"),
    ):
        fig, ax = plt.subplots(
            figsize=(8.2, 5.2)
        )

        draw_model_panel(ax, summary, model)

        ax.set_ylabel(
            "Diagnosis probe macro-F1"
        )

        fig.tight_layout()

        save(fig, out, stem)

    # CLS representation comparison across all models.
    fig, ax = plt.subplots(
        figsize=(8.7, 5.2)
    )

    for model, marker, offset in zip(
        MODELS,
        ("o", "s", "^"),
        (-0.09, 0, 0.09)
    ):
        x, y, sd = get_curve(
            summary, model, "CLS"
        )

        error_line(
            ax,
            x,
            y,
            sd,
            model,
            MODEL_COLORS[model],
            marker,
            offset
        )

    format_layer_axes(ax)

    ax.set_ylabel(
        "CLS diagnosis probe macro-F1"
    )

    ax.set_title(
        "CLS probing across RoBERTa architectures "
        "(mean ± SD)"
    )

    ax.legend(
        frameon=False,
        fontsize=9
    )

    fig.tight_layout()

    save(
        fig,
        out,
        "roberta_cls_across_models"
    )


def plot_paired_differences(all_rows, out):
    """Compare MASK minus CLS for prompted models."""

    two_models = all_rows[
        all_rows["model"].isin((DAPF, PROMPT))
    ].copy()

    paired = two_models.pivot(
        index=["model", "seed", "layer"],
        columns="representation",
        values="macro_f1"
    ).reset_index()

    paired["mask_minus_cls"] = (
        paired["MASK"] - paired["CLS"]
    )

    paired.to_csv(
        out / "roberta_mask_minus_cls_by_seed.csv",
        index=False
    )

    summary = (
        paired.groupby(
            ["model", "layer"],
            as_index=False
        )["mask_minus_cls"]
        .agg(
            mean="mean",
            sd="std",
            n_seeds="count"
        )
    )

    summary.to_csv(
        out / "roberta_mask_minus_cls_mean_sd.csv",
        index=False
    )

    fig, ax = plt.subplots(
        figsize=(8.7, 5.2)
    )

    for model, marker, offset in (
        (DAPF, "o", -0.06),
        (PROMPT, "s", 0.06)
    ):
        sub = summary[
            summary["model"] == model
        ].sort_values("layer")

        error_line(
            ax,
            sub["layer"].to_numpy(),
            sub["mean"].to_numpy(),
            sub["sd"].to_numpy(),
            model,
            MODEL_COLORS[model],
            marker,
            offset
        )

    ax.axhline(
        0,
        linestyle=":",
        color="gray",
        linewidth=1
    )

    ax.set_xticks(LAYERS)
    ax.set_xlabel("Transformer layer (1–12)")

    ax.set_ylabel(
        "MASK − CLS diagnosis probe macro-F1"
    )

    ax.set_title(
        "Within-seed representation differences "
        "(mean ± SD)"
    )

    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)

    fig.tight_layout()

    save(
        fig,
        out,
        "roberta_mask_minus_cls_by_layer"
    )


def plot_final_layer(all_rows, out):
    """Compare all five available final representations."""

    df = all_rows[
        all_rows["layer"] == 12
    ]

    order = [
        (DAPF, "CLS"),
        (DAPF, "MASK"),
        (CLS, "CLS"),
        (PROMPT, "CLS"),
        (PROMPT, "MASK"),
    ]

    agg = (
        df.groupby(
            ["model", "representation"]
        )["macro_f1"]
        .agg(
            mean="mean",
            sd="std",
            n_seeds="count"
        )
    )

    agg.reset_index().to_csv(
        out / "roberta_final_layer_mean_sd.csv",
        index=False
    )

    for key in order:
        if (
            key not in agg.index
            or int(agg.loc[key, "n_seeds"]) != 5
        ):
            raise ValueError(
                f"Missing final-layer result: {key}"
            )

    means = [
        float(agg.loc[key, "mean"])
        for key in order
    ]

    stds = [
        float(agg.loc[key, "sd"])
        for key in order
    ]

    labels = [
        "DAPF\nCLS",
        "DAPF\nMASK",
        "Plain CLS\nCLS",
        "CLS+Prompt\nCLS",
        "CLS+Prompt\naux. MASK",
    ]

    fig, ax = plt.subplots(
        figsize=(9.0, 5.3)
    )

    ax.bar(
        np.arange(5),
        means,
        color=[
            REP_COLORS["CLS"],
            REP_COLORS["MASK"],
            MODEL_COLORS[CLS],
            "#557db1",
            "#d49440",
        ],
        alpha=0.9
    )

    ax.errorbar(
        np.arange(5),
        means,
        yerr=stds,
        fmt="none",
        color="black",
        capsize=4
    )

    ax.set_xticks(
        np.arange(5),
        labels
    )

    ax.set_ylim(0, 1.02)

    ax.set_ylabel(
        "Final-layer diagnosis probe macro-F1"
    )

    ax.set_title(
        "Final-layer RoBERTa representation probes "
        "(mean ± SD)"
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.tight_layout()

    save(
        fig,
        out,
        "roberta_final_layer_probe_comparison"
    )

    print(
        "\nFinal-layer results "
        "(mean ± SD across five seeds):"
    )

    for key in order:
        print(
            f"{key[0]:22s} {key[1]:4s}: "
            f"{float(agg.loc[key, 'mean']):.4f} ± "
            f"{float(agg.loc[key, 'sd']):.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dapf-root",
        default="./output_roberta_interpretability"
    )

    parser.add_argument(
        "--cls-root",
        default="./output_roberta_cls_interpretability_multiseed"
    )

    parser.add_argument(
        "--prompt-root",
        default="./output_roberta_cls_prompt_interpretability_multiseed"
    )

    parser.add_argument(
        "--out-dir",
        default="./output_roberta_comparison/figures"
    )

    args = parser.parse_args()

    out = Path(args.out_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = pd.concat(
        [
            load_model(
                Path(args.dapf_root),
                DAPF
            ),
            load_model(
                Path(args.cls_root),
                CLS
            ),
            load_model(
                Path(args.prompt_root),
                PROMPT
            ),
        ],
        ignore_index=True
    )

    all_rows.to_csv(
        out / "roberta_layerwise_probe_by_seed.csv",
        index=False
    )

    summary = (
        all_rows.groupby(
            ["model", "layer", "representation"],
            as_index=False
        )["macro_f1"]
        .agg(
            mean="mean",
            sd="std",
            n_seeds="count"
        )
    )

    summary.to_csv(
        out / "roberta_layerwise_mean_sd.csv",
        index=False
    )

    plot_layers(summary, out)
    plot_paired_differences(all_rows, out)
    plot_final_layer(all_rows, out)

    print(
        "\nCompleted: 15 saved seed-level probing CSVs."
    )

    print(
        "No model inference or retraining performed."
    )


if __name__ == "__main__":
    main()

