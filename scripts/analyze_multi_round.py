import os
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Dictionary of algorithms to plot, mapping strategy name to its plot style configurations
algorithms_for_plot = {
    'fedavg':               {'label': 'FedAvg',              'color': '#7f7f7f', 'marker': 'o'},
    'fedprox':              {'label': 'FedProx',             'color': '#aec7e8', 'marker': 'P'},
    'scaffold':             {'label': 'SCAFFOLD',            'color': '#98df8a', 'marker': 'd'},
    'otfusion':             {'label': 'OTFusion',            'color': '#9467bd', 'marker': 'v'},
    'regmean':              {'label': 'RegMean',             'color': '#8c564b', 'marker': 'p'},
    'pfnm':                 {'label': 'PFNM',                'color': '#bcbd22', 'marker': '<'},
    'dense':                {'label': 'DENSE',               'color': '#d62728', 'marker': 'h'},
    'fisher_merge:diagonal':{'label': 'FisherMerge',         'color': '#e377c2', 'marker': '*'},
    'fedlpa':               {'label': 'FedLPA',              'color': '#1f77b4', 'marker': '>'},
    'fedfisher:diagonal':   {'label': 'FedFisher(Diagonal)', 'color': '#ff7f0e', 'marker': 's'},
    'fedfisher:kfac':       {'label': 'FedFisher(KFAC)',     'color': '#2ca02c', 'marker': '^'},
    'fedfaw:diagonal':     {'label': 'FedFAW',             'color': '#3b5998', 'marker': 'D'},
    'fedbens':              {'label': 'FedBEns',            'color': '#17becf', 'marker': 'X'},
    'fedfaw_nofisher':         {'label': 'FedFAW w/o Fisher',   'color': '#9edae5', 'marker': '8'}
}


def analyze_multi_round(paths, dataset_filter=None, datasets_filter=None, alpha_filter=None, epochs_filter=None, window=3, max_rounds=None):
    data_list = []

    # Collect data paths from directories
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue
            
        for root, dirs, files in os.walk(path):
            if "config.json" in files:
                config_path = os.path.join(root, "config.json")
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    
                    strategy = config["server"]["strategy"]
                    dataset = config["common"]["dataset"]
                    alpha = float(config["common"]["dirichlet_alpha"])
                    seed = config["common"].get("seed", "unknown")
                    epochs = config.get("client", {}).get("epochs", "unknown")

                    # Handle variants (kfac/diagonal)
                    fisher_type = config.get('fisher', {}).get('fisher_type', '')
                    if fisher_type in ['kfac', 'diagonal']:
                        if strategy in ['fedfisher', 'fedfaw', 'fisher_merge']:
                            strategy = f"{strategy}:{fisher_type}"

                    # Filters
                    if dataset_filter and dataset_filter.lower() != dataset.lower():
                        continue

                    if datasets_filter is not None and dataset.lower() not in [d.lower() for d in datasets_filter]:
                        continue

                    if alpha_filter is not None and abs(alpha - alpha_filter) > 1e-5:
                        continue

                    if epochs_filter is not None and epochs != epochs_filter:
                        continue

                    # Find all CSV files in this directory
                    csv_files = [f for f in files if f.endswith(".csv")]
                    if not csv_files:
                        continue
                    
                    csv_path = os.path.join(root, csv_files[0])
                    
                    df = pd.read_csv(csv_path)
                    if "accuracy" not in df.columns or "round" not in df.columns:
                        continue
                        
                    df = df.sort_values(by="round")
                    df["accuracy"] = df["accuracy"].rolling(window=window, min_periods=1).mean()
                        
                    for _, row in df.iterrows():
                        data_list.append({
                            "dataset": dataset,
                            "alpha": alpha,
                            "epochs": epochs,
                            "strategy": strategy,
                            "seed": seed,
                            "round": int(row["round"]),
                            "accuracy": row["accuracy"]
                        })
                except Exception as e:
                    print(f"Error processing {root}: {e}")
                    continue

    if not data_list:
        print("No valid results found matching the filters.")
        return

    df_all = pd.DataFrame(data_list)

    distinct_epochs = df_all["epochs"].unique()
    if len(distinct_epochs) > 1:
        print(f"Note: multiple client.epochs settings found in the scanned paths {sorted(distinct_epochs, key=str)}; "
              f"generating separate outputs per epochs value so they aren't pooled together as if they were more seeds. "
              f"Pass --epochs to scope to just one.")

    def format_value(mean, std):
        if pd.isna(std):
            return f"{mean:.2f}"
        return f"{mean:.2f}±{std:.2f}"

    # Group by dataset, alpha, epochs (epochs matters: different client.epochs
    # settings are different experiments, not extra seeds of the same one)
    for (dataset, alpha, epochs), gdf in df_all.groupby(["dataset", "alpha", "epochs"]):
        # Calculate mean across seeds
        agg_df = gdf.groupby(["strategy", "round"]).agg(
            mean_acc=("accuracy", "mean"),
            std_acc=("accuracy", "std")
        ).reset_index()
        
        agg_df['formatted_acc'] = agg_df.apply(lambda row: format_value(row['mean_acc'], row['std_acc']), axis=1)
        
        # Pivot table to have rounds as index and strategies as columns
        pivot_acc = agg_df.pivot(index="round", columns="strategy", values="mean_acc")
        pivot_acc.columns = [f"{c}_acc" for c in pivot_acc.columns]
        final_df = pivot_acc.reset_index()
        
        # Formatted pivot
        pivot_fmt = agg_df.pivot(index="round", columns="strategy", values="formatted_acc")
        pivot_fmt.columns = [f"{c}_acc" for c in pivot_fmt.columns]
        formatted_df = pivot_fmt.reset_index()
        
        # Reorder columns: round, strategy1_acc, strategy2_acc, ...
        cols = ["round"]
        strategies = sorted(agg_df["strategy"].unique())
        for s in strategies:
            if f"{s}_acc" in final_df.columns:
                cols.append(f"{s}_acc")
                
        final_df = final_df[cols]
        formatted_df = formatted_df[cols]
        
        out_filename = f"multi_round_summary_{dataset}_alpha{alpha}_epochs{epochs}.csv"
        final_df.to_csv(out_filename, index=False)
        print(f"Generated {out_filename}")

        out_fmt_filename = f"multi_round_formatted_summary_{dataset}_alpha{alpha}_epochs{epochs}.csv"
        formatted_df.to_csv(out_fmt_filename, index=False)
        print(f"Generated {out_fmt_filename}")

    # Plotting section
    if max_rounds is not None:
        plot_df = df_all[df_all["round"] <= max_rounds].copy()
    else:
        plot_df = df_all.copy()

    # Filter to only include algorithms in algorithms_for_plot
    plot_df = plot_df[plot_df["strategy"].isin(algorithms_for_plot.keys())]

    try:
        import seaborn as sns
        # sns.set_theme(style="darkgrid")
    except ImportError:
        pass

    # Increase global font sizes for better LaTeX rendering
    plt.rcParams.update({
        'font.size': 18, 
        'axes.titlesize': 20, 
        'axes.labelsize': 18, 
        'xtick.labelsize': 16, 
        'ytick.labelsize': 16, 
        'legend.fontsize': 17
    })

    algo_styles = algorithms_for_plot

    for (alpha, epochs), adf in plot_df.groupby(["alpha", "epochs"]):
        desired_order = ["fashion_mnist", "cifar10", "cinic10", "celeba", "cifar100"]
        datasets = sorted(adf["dataset"].unique(), key=lambda x: desired_order.index(x) if x in desired_order else 100)
        n_datasets = len(datasets)
        
        # Responsive grid layout: max 3 columns per row
        n_cols = min(3, n_datasets)
        n_rows = (n_datasets + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.5 * n_rows), sharey=False)
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
            
        for i, dataset in enumerate(datasets):
            ax = axes_flat[i]
            ddf = adf[adf["dataset"] == dataset]
            
            agg_plot = ddf.groupby(["strategy", "round"]).agg(
                mean_acc=("accuracy", "mean"),
                std_acc=("accuracy", "std")
            ).reset_index()
            
            # Sort by the order in algorithms_for_plot for consistent legend
            strategies = [alg for alg in algorithms_for_plot.keys() if alg in agg_plot["strategy"].unique()]
            
            for strategy in strategies:
                sdf = agg_plot[agg_plot["strategy"] == strategy]
                style = algo_styles[strategy]
                ax.plot(sdf["round"], sdf["mean_acc"], label=style['label'], 
                        marker=style['marker'], markersize=8, color=style['color'], linewidth=3)
                ax.fill_between(sdf["round"], sdf["mean_acc"] - sdf["std_acc"].fillna(0), 
                                sdf["mean_acc"] + sdf["std_acc"].fillna(0), 
                                color=style['color'], alpha=0.2)
                
            # Formatting to match the requested dataset names
            dataset_titles = {
                'fashion_mnist': 'FMNIST',
                'cifar10': 'CIFAR-10',
                'cinic10': 'CINIC-10',
                'celeba': 'Celeba',
                'cifar100': 'CIFAR-100'
            }
            ax.set_title(dataset_titles.get(dataset, dataset.capitalize() if dataset.islower() else dataset))
            ax.set_xlabel("Rounds")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if i % n_cols == 0:
                ax.set_ylabel("Test Accuracy")
            else:
                ax.set_ylabel("")

            ax.grid(True, linestyle='--', alpha=0.6)
            
        # Hide completely unused subplots natively
        for j in range(n_datasets, len(axes_flat)):
            fig.delaxes(axes_flat[j])
            
        plt.tight_layout()
        # tight_layout already sized the bottom margin to exactly fit the
        # x-axis labels ("Rounds") with no slack -- remember that before
        # adding the legend, so the legend's space gets added on top of it
        # rather than replacing it (replacing it was the bug: it discarded
        # the x-label's own space, so the legend ended up overlapping
        # "Rounds" whenever the figure was short, e.g. a single-row grid
        # from plotting just 3 datasets).
        tight_bottom = fig.subplotpars.bottom

        handles, labels = axes_flat[0].get_legend_handles_labels()
        ncol = min(len(labels), 5)
        # Anchor at the very bottom edge for now -- measured below and
        # added on top of tight_bottom, so it works regardless of font
        # size, number of legend rows, or figure size.
        legend = fig.legend(handles, labels, loc='lower center', ncol=ncol, bbox_to_anchor=(0.5, 0.0))

        fig.canvas.draw()
        legend_height_fig = legend.get_window_extent().transformed(fig.transFigure.inverted()).height
        bottom_fraction = min(0.5, tight_bottom + legend_height_fig + 0.02)
        plt.subplots_adjust(bottom=bottom_fraction)
        
        # Center the bottom row of subplots if there are exactly 5 datasets
        if n_datasets == 5 and n_cols == 3:
            pos0 = axes_flat[0].get_position()
            pos1 = axes_flat[1].get_position()
            shift = (pos1.x0 - pos0.x0) / 2.0
            for ax in axes_flat[3:5]:
                pos = ax.get_position()
                ax.set_position([pos.x0 + shift, pos.y0, pos.width, pos.height])
        
        plot_filename = f"accuracy_plot_alpha{alpha}_epochs{epochs}.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated plot {plot_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze FL experiment results across multiple rounds.')
    parser.add_argument('--paths', type=str, nargs='+', required=True, help='Path(s) to the results directory (e.g., out/multi-round)')
    parser.add_argument('--dataset', type=str, help='Filter by a single dataset name')
    parser.add_argument('--datasets', type=str, nargs='+', help='Filter by a set of dataset names, e.g. --datasets fashion_mnist cifar10 cinic10')
    parser.add_argument('--alpha', type=float, help='Filter by Dirichlet alpha value')
    parser.add_argument('--epochs', type=int, help='Filter by client.epochs (local training epochs per round)')
    parser.add_argument('--window', type=int, default=3, help='Rolling mean window size')
    parser.add_argument('--max_rounds', type=int, default=None, help='Max rounds to include in plot')
    args = parser.parse_args()

    analyze_multi_round(args.paths, dataset_filter=args.dataset, datasets_filter=args.datasets, alpha_filter=args.alpha, epochs_filter=args.epochs, window=args.window, max_rounds=args.max_rounds)
