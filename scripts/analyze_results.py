import os
import json
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict

def format_value(mean, std):
    if pd.isna(std):
        return f"{mean:.2f}"
    return f"{mean:.2f}±{std:.2f}"

def build_report_df(paths, strategy_filter=None, dataset_filter=None, alpha_filter=None, alphas_filter=None, val_samples_filter=None, max_rounds=1):
    """
    Walks `paths` for config.json + result CSVs, applies the given filters,
    and returns a long-format DataFrame with one row per
    (Algorithm, Dataset, Alpha, Val_Samples), aggregated across seeds.

    max_rounds truncates each run's CSV to round <= max_rounds before
    aggregating (accuracy, time, upload/download all computed from that
    truncated range) -- defaults to 1 since this script is for single-round
    comparisons; pass a higher value (or None for no cutoff) if pointing it
    at multi-round data.

    Shared by analyze_results.py's CLI report and analyze_results_table.py's
    pivoted paper-table view, so both read from the exact same aggregation
    logic instead of two copies that could drift apart.
    """
    groups = {
        "strategy": [],
        "dataset": [],
        "dirichlet_alpha": [],
        "val_samples": [],
        "seed": [],
        "path": [],
    }

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
                    val_samples = config.get("fisher", {}).get("val_samples", "N/A")
                    
                    # Handle variants (kfac/diagonal)
                    fisher_type = config.get('fisher', {}).get('fisher_type', '')
                    if fisher_type in ['kfac', 'diagonal']:
                        if strategy in ['fedfisher', 'fedfaw', 'fisher_merge']:
                            strategy = f"{strategy}:{fisher_type}"

                    # Filters
                    if strategy_filter and strategy_filter.lower() not in strategy.lower():
                        continue
                    if dataset_filter and dataset_filter.lower() != dataset.lower():
                        continue

                    if alpha_filter is not None and abs(alpha - alpha_filter) > 1e-5:
                        continue

                    if alphas_filter is not None and not any(abs(alpha - a) < 1e-5 for a in alphas_filter):
                        continue

                    if val_samples_filter is not None and str(val_samples) != str(val_samples_filter):
                        continue
                    
                    # Find all CSV files in this directory
                    csv_files = [f for f in files if f.endswith(".csv")]
                    if not csv_files:
                        continue
                    
                    csv_path = os.path.join(root, csv_files[0])
                    
                    groups["strategy"].append(strategy)
                    groups["dataset"].append(dataset)
                    groups["dirichlet_alpha"].append(alpha)
                    groups["val_samples"].append(val_samples)
                    groups["seed"].append(seed)
                    groups["path"].append(csv_path)
                except Exception as e:
                    continue

    df_meta = pd.DataFrame(groups)
    if df_meta.empty:
        print("No valid results found matching the filters.")
        return pd.DataFrame()

    report = []

    # Group by experimental settings
    for (strategy, dataset, alpha, val_samples), gdf in df_meta.groupby(["strategy", "dataset", "dirichlet_alpha", "val_samples"]):
        all_dfs = []
        seeds_found = sorted(gdf["seed"].unique())
        
        for csv_path in gdf["path"]:
            try:
                temp_df = pd.read_csv(csv_path)
                if max_rounds is not None and "round" in temp_df.columns:
                    temp_df = temp_df[temp_df["round"] <= max_rounds]
                if not temp_df.empty and "accuracy" in temp_df.columns:
                    all_dfs.append(temp_df)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")

        if not all_dfs:
            continue

        # Aggregate across seeds
        seed_metrics = []
        for df in all_dfs:
            upload = df["uploaded_mib"].sum() if "uploaded_mib" in df.columns else 0
            download = df["downloaded_mib"].sum() if "downloaded_mib" in df.columns else 0
            metrics = {
                "accuracy": df["accuracy"].iloc[-1],
                "time": df["time"].sum() if "time" in df.columns else 0,
                "upload": upload,
                "download": download,
                "total": upload + download,
                "round": df["round"].max()
            }
            seed_metrics.append(metrics)
        
        sm_df = pd.DataFrame(seed_metrics)
        
        acc_m, acc_s = sm_df["accuracy"].mean(), sm_df["accuracy"].std()
        time_m, time_s = sm_df["time"].mean(), sm_df["time"].std()
        up_m, up_s = sm_df["upload"].mean(), sm_df["upload"].std()
        down_m, down_s = sm_df["download"].mean(), sm_df["download"].std()
        total_m, total_s = sm_df["total"].mean(), sm_df["total"].std()
        max_round = sm_df["round"].max()

        report.append({
            "Algorithm": strategy,
            "Dataset": dataset,
            "Alpha": alpha,
            "Val_Samples": val_samples,
            "Last Acc": format_value(acc_m, acc_s),
            "Time (s)": format_value(time_m, time_s),
            "Upload (MiB)": format_value(up_m, up_s),
            "Download (MiB)": format_value(down_m, down_s),
            "Total (MiB)": format_value(total_m, total_s),
            "Mean_Val": acc_m, # for sorting
            "Round": int(max_round),
            "Seeds": ", ".join(map(str, seeds_found)),
            "Num_Seeds": len(all_dfs)
        })

    return pd.DataFrame(report)


def analyze_results(paths, strategy_filter=None, dataset_filter=None, alpha_filter=None, alphas_filter=None, val_samples_filter=None, max_rounds=1):
    df_report = build_report_df(paths, strategy_filter=strategy_filter, dataset_filter=dataset_filter,
                                alpha_filter=alpha_filter, alphas_filter=alphas_filter,
                                val_samples_filter=val_samples_filter, max_rounds=max_rounds)
    if df_report.empty:
        return

    # Calculate column widths
    max_algo_len = max([len(a) for a in df_report['Algorithm']] + [len('Algorithm')])

    # Print the requested columns
    header_line = f"{'Algorithm':<{max_algo_len}} | {'Dataset':<10} | {'Alpha':<6} | {'Val Smp':<8} | {'Acc (%)':<12} | {'Time (s)':<12} | {'Up (MiB)':<10} | {'Down (MiB)':<10} | {'Total (MiB)':<12} | {'R'}"
    separator = f"{'-'*max_algo_len}-|-{'-'*10}-|-{'-'*6}-|-{'-'*8}-|-{'-'*12}-|-{'-'*12}-|-{'-'*10}-|-{'-'*10}-|-{'-'*12}-|-{'-'*2}"
    
    print(f"\n{'='*len(separator)}")
    print(header_line)
    print(separator)
    
    # Handle numeric string coercion structurally to sort correctly alongside missing vals
    df_report["Val_Samples_Sort"] = pd.to_numeric(df_report["Val_Samples"], errors='coerce').fillna(-1)
    df_sorted = df_report.sort_values(by=["Dataset", "Alpha", "Val_Samples_Sort", "Mean_Val"], ascending=[True, True, True, False])
    df_sorted = df_sorted.drop(columns=["Val_Samples_Sort"])
    
    for _, row in df_sorted.iterrows():
        print(f"{row['Algorithm']:<{max_algo_len}} | {row['Dataset']:<10} | {row['Alpha']:<6} | {str(row['Val_Samples']):<8} | {row['Last Acc']:<12} | {row['Time (s)']:<12} | {row['Upload (MiB)']:<10} | {row['Download (MiB)']:<10} | {row['Total (MiB)']:<12} | {row['Round']}")

    # Save to CSV
    df_sorted.drop(columns=['Mean_Val']).to_csv("summary_results.csv", index=False)
    print(f"\nDetailed summary saved to summary_results.csv")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze FL experiment results.')
    parser.add_argument('--paths', type=str, nargs='+', help='Path(s) to the results directory (e.g., out/)')
    parser.add_argument('--strategy', type=str, help='Filter by strategy name')
    parser.add_argument('--dataset', type=str, help='Filter by dataset name')
    parser.add_argument('--alpha', type=float, help='Filter by a single Dirichlet alpha value')
    parser.add_argument('--alphas', type=float, nargs='+', help='Filter by a set of Dirichlet alpha values (e.g. --alphas 0.05 0.1)')
    parser.add_argument('--val_samples', type=str, help='Filter by val_samples')
    parser.add_argument('--max_rounds', type=int, default=1, help='Truncate each run to round <= max_rounds before aggregating (default 1, for single-round comparisons; pass a higher value for multi-round data)')
    args = parser.parse_args()

    analyze_results(args.paths, strategy_filter=args.strategy, dataset_filter=args.dataset, alpha_filter=args.alpha, alphas_filter=args.alphas, val_samples_filter=args.val_samples, max_rounds=args.max_rounds)

