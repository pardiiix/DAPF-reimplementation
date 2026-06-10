import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


LABEL_NAMES = {
    0: "Healthy",
    1: "Dementia",
}


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_inputs(results_csv, interpretability_pt):
    results_df = pd.read_csv(results_csv)
    obj = torch.load(interpretability_pt, map_location="cpu")

    required_pt_keys = {"ids", "cls_hidden_states", "mask_hidden_states"}
    missing = required_pt_keys - set(obj.keys())

    if missing:
        raise ValueError(f"Missing keys in interpretability file: {missing}")

    required_result_cols = {"id", "labels", "pred_labels", "probas"}
    missing_cols = required_result_cols - set(results_df.columns)

    if missing_cols:
        raise ValueError(f"Missing columns in results CSV: {missing_cols}")

    ids = list(obj["ids"])

    cls_hidden = tensor_to_numpy(obj["cls_hidden_states"])
    mask_hidden = tensor_to_numpy(obj["mask_hidden_states"])

    meta_df = pd.DataFrame({"id": ids})
    meta_df = meta_df.merge(
        results_df[["id", "labels", "pred_labels", "probas"]],
        on="id",
        how="left",
    )

    if meta_df["labels"].isna().any():
        missing_ids = meta_df.loc[meta_df["labels"].isna(), "id"].tolist()
        raise ValueError(f"Some IDs from the PT file were not found in results CSV: {missing_ids[:10]}")

    meta_df["labels"] = meta_df["labels"].astype(int)
    meta_df["pred_labels"] = meta_df["pred_labels"].astype(int)
    meta_df["probas"] = meta_df["probas"].astype(float)

    meta_df["diagnosis_label"] = meta_df["labels"].map(LABEL_NAMES)
    meta_df["predicted_label"] = meta_df["pred_labels"].map(LABEL_NAMES)
    meta_df["is_correct"] = meta_df["labels"] == meta_df["pred_labels"]
    meta_df["correctness_label"] = meta_df["is_correct"].map({
        True: "Correct",
        False: "Incorrect",
    })

    if cls_hidden.shape[0] != len(meta_df):
        raise ValueError(
            f"CLS hidden state count {cls_hidden.shape[0]} does not match metadata count {len(meta_df)}"
        )

    if mask_hidden.shape[0] != len(meta_df):
        raise ValueError(
            f"MASK hidden state count {mask_hidden.shape[0]} does not match metadata count {len(meta_df)}"
        )

    return meta_df, cls_hidden, mask_hidden


def standardize_hidden_states(x):
    return StandardScaler().fit_transform(x)


def run_pca(x, random_state):
    reducer = PCA(n_components=2, random_state=random_state)
    return reducer.fit_transform(x)


def run_tsne(x, random_state, perplexity):
    n = x.shape[0]

    if perplexity >= n:
        adjusted = max(2, min(10, n // 3))
        warnings.warn(
            f"t-SNE perplexity={perplexity} is too large for n={n}. "
            f"Using perplexity={adjusted} instead."
        )
        perplexity = adjusted

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    )

    return reducer.fit_transform(x)


def run_umap(x, random_state, n_neighbors, min_dist):
    if not HAS_UMAP:
        return None

    n = x.shape[0]
    n_neighbors = min(n_neighbors, n - 1)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )

    return reducer.fit_transform(x)


def compute_silhouette(x, labels):
    labels = np.asarray(labels)

    if len(np.unique(labels)) < 2:
        return np.nan

    try:
        return float(silhouette_score(x, labels))
    except Exception:
        return np.nan


def plot_projection(
    projection,
    meta_df,
    color_by,
    title,
    output_path,
):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))

    if color_by == "diagnosis":
        column = "diagnosis_label"
        order = ["Healthy", "Dementia"]
    elif color_by == "correctness":
        column = "correctness_label"
        order = ["Correct", "Incorrect"]
    else:
        raise ValueError("color_by must be either 'diagnosis' or 'correctness'.")

    for group in order:
        mask = meta_df[column] == group

        if mask.sum() == 0:
            continue

        ax.scatter(
            projection[mask, 0],
            projection[mask, 1],
            label=f"{group} (n={mask.sum()})",
            alpha=0.85,
            s=55,
        )

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_projection_csv(
    projection,
    meta_df,
    representation_name,
    method_name,
    output_path,
):
    out = meta_df.copy()
    out["representation"] = representation_name
    out["method"] = method_name
    out["dim1"] = projection[:, 0]
    out["dim2"] = projection[:, 1]
    out.to_csv(output_path, index=False)


def run_visualizations(args):
    os.makedirs(args.output_dir, exist_ok=True)

    meta_df, cls_hidden, mask_hidden = load_inputs(
        results_csv=args.results_csv,
        interpretability_pt=args.interpretability_pt,
    )

    representations = {
        "cls": cls_hidden,
        "mask": mask_hidden,
    }

    plot_specs = [
        {
            "representation": "cls",
            "color_by": "diagnosis",
            "title": "[CLS] hidden states colored by true diagnosis",
        },
        {
            "representation": "mask",
            "color_by": "diagnosis",
            "title": "[MASK] hidden states colored by true diagnosis",
        },
        {
            "representation": "mask",
            "color_by": "correctness",
            "title": "[MASK] hidden states colored by prediction correctness",
        },
    ]

    metrics_rows = []
    all_projection_rows = []

    for representation_name, hidden_states in representations.items():
        x = standardize_hidden_states(hidden_states)

        highdim_diag_silhouette = compute_silhouette(
            x,
            meta_df["labels"].to_numpy(),
        )

        highdim_correct_silhouette = compute_silhouette(
            x,
            meta_df["is_correct"].astype(int).to_numpy(),
        )

        metrics_rows.append({
            "representation": representation_name,
            "method": "high_dimensional",
            "color_by": "diagnosis",
            "silhouette": highdim_diag_silhouette,
        })

        metrics_rows.append({
            "representation": representation_name,
            "method": "high_dimensional",
            "color_by": "correctness",
            "silhouette": highdim_correct_silhouette,
        })

        projections = {}

        projections["pca"] = run_pca(x, random_state=args.seed)
        projections["tsne"] = run_tsne(
            x,
            random_state=args.seed,
            perplexity=args.tsne_perplexity,
        )

        if HAS_UMAP:
            projections["umap"] = run_umap(
                x,
                random_state=args.seed,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
            )
        else:
            print("UMAP is not installed. Skipping UMAP plots. Install with: pip install umap-learn")

        for method_name, projection in projections.items():
            if projection is None:
                continue

            projection_csv = os.path.join(
                args.output_dir,
                f"{representation_name}_{method_name}_projection.csv",
            )

            save_projection_csv(
                projection=projection,
                meta_df=meta_df,
                representation_name=representation_name,
                method_name=method_name,
                output_path=projection_csv,
            )

            temp = meta_df.copy()
            temp["representation"] = representation_name
            temp["method"] = method_name
            temp["dim1"] = projection[:, 0]
            temp["dim2"] = projection[:, 1]
            all_projection_rows.append(temp)

            metrics_rows.append({
                "representation": representation_name,
                "method": method_name,
                "color_by": "diagnosis",
                "silhouette": compute_silhouette(
                    projection,
                    meta_df["labels"].to_numpy(),
                ),
            })

            metrics_rows.append({
                "representation": representation_name,
                "method": method_name,
                "color_by": "correctness",
                "silhouette": compute_silhouette(
                    projection,
                    meta_df["is_correct"].astype(int).to_numpy(),
                ),
            })

    for spec in plot_specs:
        representation_name = spec["representation"]
        hidden_states = representations[representation_name]
        x = standardize_hidden_states(hidden_states)

        methods = {
            "pca": run_pca(x, random_state=args.seed),
            "tsne": run_tsne(
                x,
                random_state=args.seed,
                perplexity=args.tsne_perplexity,
            ),
        }

        if HAS_UMAP:
            methods["umap"] = run_umap(
                x,
                random_state=args.seed,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
            )

        for method_name, projection in methods.items():
            output_png = os.path.join(
                args.output_dir,
                f"{representation_name}_{spec['color_by']}_{method_name}.png",
            )

            title = f"{spec['title']} ({method_name.upper()})"

            plot_projection(
                projection=projection,
                meta_df=meta_df,
                color_by=spec["color_by"],
                title=title,
                output_path=output_png,
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(args.output_dir, "representation_visualization_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    if all_projection_rows:
        all_projection_df = pd.concat(all_projection_rows, ignore_index=True)
        all_projection_path = os.path.join(args.output_dir, "all_representation_projections.csv")
        all_projection_df.to_csv(all_projection_path, index=False)

    print(f"Saved representation visualizations to: {args.output_dir}")
    print(f"Saved visualization metrics to: {metrics_path}")
    print("\nGenerated plots:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith(".png"):
            print(f"  {f}")

    print("\nVisualization metrics:")
    print(metrics_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results_csv", required=True)
    parser.add_argument("--interpretability_pt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tsne_perplexity", type=float, default=10.0)
    parser.add_argument("--umap_neighbors", type=int, default=10)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)

    args = parser.parse_args()

    run_visualizations(args)


if __name__ == "__main__":
    main()