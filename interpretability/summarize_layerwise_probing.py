import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_seed_results(interp_root, seeds):
    rows = []

    for seed in seeds:
        path = os.path.join(
            interp_root,
            f"seed_{seed}",
            "layerwise_probing",
            "layerwise_probe_results.csv",
        )

        if not os.path.exists(path):
            print(f"Warning: missing layerwise file for seed {seed}: {path}")
            continue

        df = pd.read_csv(path)
        df["seed"] = seed
        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def summarize(all_df):
    summary = (
        all_df
        .groupby(["layer", "representation"])
        .agg(
            mean_macro_f1=("score_mean", "mean"),
            std_macro_f1=("score_mean", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    summary["mean_pm_std"] = summary.apply(
        lambda r: f"{r['mean_macro_f1']:.4f} ± {r['std_macro_f1']:.4f}",
        axis=1,
    )

    return summary


def plot_summary(summary_df, output_path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for representation in ["CLS", "MASK"]:
        sub = summary_df[summary_df["representation"] == representation].copy()
        sub = sub.sort_values("layer")

        ax.plot(
            sub["layer"],
            sub["mean_macro_f1"],
            marker="o",
            label=representation,
        )

        ax.fill_between(
            sub["layer"],
            sub["mean_macro_f1"] - sub["std_macro_f1"],
            sub["mean_macro_f1"] + sub["std_macro_f1"],
            alpha=0.18,
        )

    ax.set_xlabel("BERT layer")
    ax.set_ylabel("Probe macro-F1")
    ax.set_title("Layer-wise diagnosis probing across seeds")
    ax.set_xticks(sorted(summary_df["layer"].unique()))
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--interp_root", default="./output_interpretability_multiseed")
    parser.add_argument(
        "--output_dir",
        default="./output_interpretability_multiseed_summary/layerwise_probing",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")

    args = parser.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]

    os.makedirs(args.output_dir, exist_ok=True)

    all_df = load_seed_results(args.interp_root, seeds)

    if all_df.empty:
        raise ValueError("No layer-wise probing results found.")

    summary_df = summarize(all_df)

    all_path = os.path.join(args.output_dir, "all_seed_layerwise_probe_results.csv")
    summary_path = os.path.join(args.output_dir, "layerwise_probe_summary.csv")
    plot_path = os.path.join(args.output_dir, "layerwise_probe_summary.png")

    all_df.to_csv(all_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    plot_summary(summary_df, plot_path)

    print("Saved:")
    print(all_path)
    print(summary_path)
    print(plot_path)

    print("\nLayer-wise summary:")
    print(
        summary_df
        .pivot(index="layer", columns="representation", values="mean_pm_std")
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
