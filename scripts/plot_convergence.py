"""Convergence plots (first N rounds) in the same style as analyze_multi_round.py.

Strategies found under --override_paths replace the same strategy from --paths,
so re-run (bug-fixed) methods can be swapped in without deleting old runs, e.g.

python scripts/plot_convergence.py \
    --paths out/multi-round \
    --override_paths out/multi-round-rerun \
    --datasets fashion_mnist cifar10 cinic10 --alpha 0.1 --epochs 10 --max_rounds 5
"""
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_multi_round import algorithms_for_plot  # same labels/colors/markers

DATASET_TITLES = {
    'fashion_mnist': 'FMNIST',
    'cifar10': 'CIFAR-10',
    'cinic10': 'CINIC-10',
    'celeba': 'Celeba',
    'cifar100': 'CIFAR-100',
}
DESIRED_ORDER = ["fashion_mnist", "cifar10", "cinic10", "celeba", "cifar100"]


def load_runs(paths, datasets=None, alpha=None, epochs=None, window=3):
    """Same parsing/filtering/smoothing as analyze_multi_round.analyze_multi_round."""
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue
        for root, _, files in os.walk(path):
            if "config.json" not in files:
                continue
            try:
                with open(os.path.join(root, "config.json")) as f:
                    config = json.load(f)
                strategy = config["server"]["strategy"]
                dataset = config["common"]["dataset"]
                a = float(config["common"]["dirichlet_alpha"])
                seed = config["common"].get("seed", "unknown")
                ep = config.get("client", {}).get("epochs", "unknown")

                fisher_type = config.get('fisher', {}).get('fisher_type', '')
                if fisher_type in ['kfac', 'diagonal'] and strategy in [
                        'fedfisher', 'fedfaw', 'fisher_merge']:
                    strategy = f"{strategy}:{fisher_type}"

                if datasets is not None and dataset.lower() not in [d.lower() for d in datasets]:
                    continue
                if alpha is not None and abs(a - alpha) > 1e-5:
                    continue
                if epochs is not None and ep != epochs:
                    continue

                csv_files = [f for f in files if f.endswith(".csv")]
                if not csv_files:
                    continue
                df = pd.read_csv(os.path.join(root, csv_files[0]))
                if "accuracy" not in df.columns or "round" not in df.columns:
                    continue
                df = df.sort_values(by="round")
                df["accuracy"] = df["accuracy"].rolling(window=window, min_periods=1).mean()
                for _, r in df.iterrows():
                    rows.append({"dataset": dataset, "alpha": a, "epochs": ep, "strategy": strategy,
                                 "seed": seed, "round": int(r["round"]), "accuracy": r["accuracy"]})
            except Exception as e:
                print(f"Error processing {root}: {e}")
    return pd.DataFrame(rows)


def plot_convergence(df_all, max_rounds, out_prefix="convergence_plot"):
    plot_df = df_all[df_all["round"] <= max_rounds]
    plot_df = plot_df[plot_df["strategy"].isin(algorithms_for_plot.keys())]

    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 17
    })

    for (alpha, epochs), adf in plot_df.groupby(["alpha", "epochs"]):
        datasets = sorted(adf["dataset"].unique(),
                          key=lambda x: DESIRED_ORDER.index(x) if x in DESIRED_ORDER else 100)
        n_datasets = len(datasets)
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.5 * n_rows), sharey=False)
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, dataset in enumerate(datasets):
            ax = axes_flat[i]
            agg = adf[adf["dataset"] == dataset].groupby(["strategy", "round"]).agg(
                mean_acc=("accuracy", "mean"), std_acc=("accuracy", "std")).reset_index()

            for strategy in [s for s in algorithms_for_plot if s in agg["strategy"].unique()]:
                sdf = agg[agg["strategy"] == strategy]
                style = algorithms_for_plot[strategy]
                ax.plot(sdf["round"], sdf["mean_acc"], label=style['label'],
                        marker=style['marker'], markersize=8, color=style['color'], linewidth=3)
                ax.fill_between(sdf["round"], sdf["mean_acc"] - sdf["std_acc"].fillna(0),
                                sdf["mean_acc"] + sdf["std_acc"].fillna(0),
                                color=style['color'], alpha=0.2)

            ax.set_title(DATASET_TITLES.get(dataset, dataset.capitalize()))
            ax.set_xlabel("Rounds")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_ylabel("Test Accuracy" if i % n_cols == 0 else "")
            ax.grid(True, linestyle='--', alpha=0.6)

        for j in range(n_datasets, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        # Legend below the grid, sized on top of tight_layout's bottom margin
        # (same approach as analyze_multi_round.py, so it never overlaps "Rounds").
        plt.tight_layout()
        tight_bottom = fig.subplotpars.bottom
        # Collect handles across all panels so a method missing from panel 0 still appears.
        seen = {}
        for ax in axes_flat[:n_datasets]:
            for h, l in zip(*ax.get_legend_handles_labels()):
                seen.setdefault(l, h)
        order = [v['label'] for v in algorithms_for_plot.values() if v['label'] in seen]
        legend = fig.legend([seen[l] for l in order], order, loc='lower center',
                            ncol=min(len(order), 5), bbox_to_anchor=(0.5, 0.0))
        fig.canvas.draw()
        legend_h = legend.get_window_extent().transformed(fig.transFigure.inverted()).height
        plt.subplots_adjust(bottom=min(0.5, tight_bottom + legend_h + 0.02))

        if n_datasets == 5 and n_cols == 3:
            shift = (axes_flat[1].get_position().x0 - axes_flat[0].get_position().x0) / 2.0
            for ax in axes_flat[3:5]:
                p = ax.get_position()
                ax.set_position([p.x0 + shift, p.y0, p.width, p.height])

        fname = f"{out_prefix}_alpha{alpha}_epochs{epochs}_{max_rounds}rounds"
        for ext in ("png", "pdf"):
            plt.savefig(f"{fname}.{ext}", bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated plot {fname}.png / .pdf")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description='Convergence plots for the first N rounds.')
    p.add_argument('--paths', nargs='+', required=True, help='Base results directories')
    p.add_argument('--override_paths', nargs='*', default=[],
                   help='Directories whose strategies replace the same strategy from --paths (e.g. bug-fixed re-runs)')
    p.add_argument('--datasets', nargs='+', help='e.g. --datasets fashion_mnist cifar10 cinic10')
    p.add_argument('--alpha', type=float)
    p.add_argument('--epochs', type=int, help='Filter by client.epochs')
    p.add_argument('--window', type=int, default=3, help='Rolling mean window size')
    p.add_argument('--max_rounds', type=int, default=5)
    p.add_argument('--out_prefix', default='convergence_plot')
    args = p.parse_args()

    base = load_runs(args.paths, args.datasets, args.alpha, args.epochs, args.window)
    over = load_runs(args.override_paths, args.datasets, args.alpha, args.epochs, args.window) \
        if args.override_paths else pd.DataFrame(columns=base.columns)
    if not over.empty:
        key = ["dataset", "alpha", "epochs", "strategy"]
        replaced = over[key].drop_duplicates()
        base = base.merge(replaced, on=key, how="left", indicator=True)
        base = base[base["_merge"] == "left_only"].drop(columns="_merge")
        print("Overridden from --override_paths:",
              sorted(set(map(tuple, replaced[["dataset", "strategy"]].values))))
    df_all = pd.concat([base, over], ignore_index=True)
    if df_all.empty:
        sys.exit("No valid results found matching the filters.")

    # Seeds per (dataset, strategy) as a sanity check
    print(df_all[df_all["round"] == 1].groupby(["dataset", "strategy"])["seed"].nunique().unstack(0))
    plot_convergence(df_all, args.max_rounds, args.out_prefix)
