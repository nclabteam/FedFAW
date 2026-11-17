"""
Pivots FL experiment results into a paper-style summary table: one row per
(Dataset, Partition/alpha), one column per algorithm, cells = "mean±std"
accuracy. Reuses analyze_results.py's build_report_df() for data collection
so both scripts aggregate results identically.
"""
import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_results import build_report_df

# Canonical dataset display order/names. Anything found in the data but not
# listed here is appended at the end, sorted alphabetically.
DATASET_ORDER = [
    ("fashion_mnist", "FMNIST"),
    ("cifar10", "CIFAR-10"),
    ("cinic10", "CINIC-10"),
    ("svhn", "SVHN"),
    ("stl", "STL-10"),
    ("uci_har", "UCI-HAR"),
    ("capture24", "Capture-24"),
     ("cifar100", "CIFAR-100"),
]

# Canonical algorithm columns, in display order. Each entry lists every
# internal "Algorithm" string (as produced by build_report_df, i.e. already
# suffixed with ":diagonal"/":kfac" where applicable) that should map to
# that column. Anything found in the data but not listed here is appended
# at the end, sorted alphabetically, using its raw internal name.
ALGO_ORDER = [
    ("FedAVG", ["fedavg"]),
    # ("FedProx", ["fedprox"]),
    # ("SCAFFOLD", ["scaffold"]),
    ("OTFusion", ["otfusion"]),
    ("RegMean", ["regmean"]),
    # ("PFNM", ["pfnm"]),
    ("DENSE", ["dense"]),
    ("Fisher Merge", ["fisher_merge", "fisher_merge:diagonal", "fisher_merge:kfac"]),
    ("FedLPA", ["fedlpa"]),
    ("FedFisher:DIAG", ["fedfisher:diagonal"]),
    ("FedFisher:KFAC", ["fedfisher:kfac"]),
    ("FedBEns", ["fedbens"]),
    ("FedFAW", ["fedfaw:diagonal", "fedfaw"]),
    ("FedFAW w/o fisher", ["fedfaw_nofisher", "bo_averaging"]),
]


def _resolve_duplicates(df_report):
    """
    build_report_df() groups by (Algorithm, Dataset, Alpha, Val_Samples), so
    without a --val_samples filter, more than one Val_Samples setting can
    map to the same (Dataset, Alpha, Algorithm) pivot cell. When that
    happens, keep the row backed by the most seeds and warn once so the
    ambiguity isn't silently hidden.
    """
    if df_report["Val_Samples"].nunique() <= 1:
        return df_report

    key_cols = ["Algorithm", "Dataset", "Alpha"]
    dupe_mask = df_report.duplicated(subset=key_cols, keep=False)
    if dupe_mask.any():
        distinct_vs = sorted(df_report["Val_Samples"].unique(), key=str)
        print(f"Warning: multiple val_samples settings found {distinct_vs}; "
              f"picking the one with the most seeds per cell. "
              f"Pass --val_samples to disambiguate.")

    return (df_report
            .sort_values("Num_Seeds", ascending=False)
            .drop_duplicates(subset=key_cols, keep="first"))


def build_pivot(df_report):
    df_report = _resolve_duplicates(df_report)

    # Map internal Algorithm -> display column, preserving canonical order
    # and appending any unmapped algorithms (sorted) at the end.
    algo_display = {}
    for display, keys in ALGO_ORDER:
        for k in keys:
            algo_display[k] = display
    known_algo_cols = [display for display, _ in ALGO_ORDER]
    extra_algo_cols = sorted({a for a in df_report["Algorithm"].unique() if a not in algo_display})
    algo_columns = known_algo_cols + extra_algo_cols
    df_report = df_report.copy()
    df_report["Algo_Display"] = df_report["Algorithm"].map(lambda a: algo_display.get(a, a))

    # Map internal Dataset -> display name, same pattern.
    dataset_display = dict(DATASET_ORDER)
    known_dataset_keys = [k for k, _ in DATASET_ORDER]
    extra_dataset_keys = sorted(set(df_report["Dataset"].unique()) - set(known_dataset_keys))
    dataset_order = known_dataset_keys + extra_dataset_keys
    df_report["Dataset_Display"] = df_report["Dataset"].map(lambda d: dataset_display.get(d, d))

    # Cell lookup: (dataset key, alpha, algo display) -> "mean±std" string
    cell = {}
    for _, row in df_report.iterrows():
        cell[(row["Dataset"], row["Alpha"], row["Algo_Display"])] = row["Last Acc"]

    rows = []
    for dataset_key in dataset_order:
        alphas = sorted(df_report.loc[df_report["Dataset"] == dataset_key, "Alpha"].unique())
        for alpha in alphas:
            row = {
                "Dataset": dataset_display.get(dataset_key, dataset_key),
                "Partition": f"{alpha:g}",
            }
            for algo in algo_columns:
                row[algo] = cell.get((dataset_key, alpha, algo), "")
            rows.append(row)

    return pd.DataFrame(rows, columns=["Dataset", "Partition"] + algo_columns)


def print_table(df_pivot):
    cols = list(df_pivot.columns)
    widths = {c: max(len(c), df_pivot[c].astype(str).map(len).max() if not df_pivot.empty else 0) for c in cols}

    def fmt_row(values):
        return " | ".join(f"{str(v):<{widths[c]}}" for c, v in zip(cols, values))

    header = fmt_row(cols)
    separator = "-+-".join("-" * widths[c] for c in cols)

    print(f"\n{'=' * len(separator)}")
    print(header)
    print(separator)

    last_dataset = None
    for _, row in df_pivot.iterrows():
        display_row = row.copy()
        if display_row["Dataset"] == last_dataset:
            display_row["Dataset"] = ""
        else:
            last_dataset = row["Dataset"]
        print(fmt_row(display_row[c] for c in cols))


def analyze_results_table(paths, strategy_filter=None, dataset_filter=None, alpha_filter=None,
                          alphas_filter=None, val_samples_filter=None, output="summary_results_table.csv"):
    df_report = build_report_df(paths, strategy_filter=strategy_filter, dataset_filter=dataset_filter,
                                alpha_filter=alpha_filter, alphas_filter=alphas_filter,
                                val_samples_filter=val_samples_filter)
    if df_report.empty:
        return

    df_pivot = build_pivot(df_report)
    print_table(df_pivot)

    df_pivot.to_csv(output, index=False)
    print(f"\nPivoted summary table saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pivot FL experiment results into a paper-style summary table.')
    parser.add_argument('--paths', type=str, nargs='+', required=True, help='Path(s) to the results directory (e.g., out/)')
    parser.add_argument('--strategy', type=str, help='Filter by strategy name')
    parser.add_argument('--dataset', type=str, help='Filter by dataset name')
    parser.add_argument('--alpha', type=float, help='Filter by a single Dirichlet alpha value')
    parser.add_argument('--alphas', type=float, nargs='+', help='Filter by a set of Dirichlet alpha values (e.g. --alphas 0.05 0.1)')
    parser.add_argument('--val_samples', type=str, help='Filter by val_samples')
    parser.add_argument('--output', type=str, default='summary_results_table.csv', help='Output CSV path')
    args = parser.parse_args()

    analyze_results_table(args.paths, strategy_filter=args.strategy, dataset_filter=args.dataset,
                          alpha_filter=args.alpha, alphas_filter=args.alphas,
                          val_samples_filter=args.val_samples, output=args.output)
