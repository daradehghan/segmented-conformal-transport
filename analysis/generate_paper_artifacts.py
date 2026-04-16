"""Generate the paper-facing tables and figures from saved benchmark results.

This is the manuscript-facing artifact path for the real-data figures and
tables. It is authoritative only when the input real-data snapshot comes from
the synchronized post-``n_cal`` benchmark runner. Legacy real-data snapshots
remain loadable for backward compatibility, but they are not manuscript
authoritative.

Outputs (default: analysis/output/paper):
    tables/dataset_summary.csv
    tables/headline_real_data.csv
    tables/coverage_variability.csv
    tables/paired_wilcoxon.csv
    tables/real_data_runtime_summary.csv
    tables/synthetic_representative_summary.csv
    tables/synthetic_representative_diagnostics.csv
    tables/synthetic_full_suite_part1.csv
    tables/synthetic_full_suite_part2.csv
    figures/median_coverage_vs_width.pdf
    figures/median_coverage_vs_width.png
    figures/eseries_distribution.pdf
    figures/eseries_distribution.png
    figures/coverage_distribution.pdf
    figures/coverage_distribution.png
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATASET_LABELS = {
    "electricity_cached": "Electricity Hourly",
    "electricity": "Electricity Hourly",
    "fred_md": "FRED-MD",
    "traffic": "Traffic Hourly",
}

DATASET_ORDER = ["electricity_cached", "electricity", "fred_md", "traffic"]
METHOD_ORDER = ["SCT", "ACI", "split-CP"]

METHOD_COLOURS = {
    "SCT": "#2166ac",
    "ACI": "#b2182b",
    "split-CP": "#4daf4a",
}

METHOD_MARKERS = {
    "SCT": "o",
    "ACI": "s",
    "split-CP": "^",
}

PAPER_TABLE_FILES = (
    "dataset_summary.csv",
    "headline_real_data.csv",
    "coverage_variability.csv",
    "paired_wilcoxon.csv",
    "real_data_runtime_summary.csv",
)

SYNTHETIC_TABLE_FILES = (
    "synthetic_representative_summary.csv",
    "synthetic_representative_diagnostics.csv",
    "synthetic_full_suite_part1.csv",
    "synthetic_full_suite_part2.csv",
)

PAPER_FIGURE_FILES = (
    "median_coverage_vs_width.png",
    "eseries_distribution.png",
    "coverage_distribution.png",
)

SYNTHETIC_SCENARIO_ORDER = [
    "A: R=3, L=500, phi=0.0",
    "A: R=3, L=500, phi=0.5",
    "B: L=1000, λ=0.0",
    "B: L=1000, λ=0.2",
    "B: L=1000, λ=0.4",
    "C: L=500, Δb=0.5",
    "C: L=500, Δb=1.0",
    "D: L_short=25",
    "D: L_short=100",
    "F: ν=3",
    "F: ν=5",
    "G: γ=0.0",
    "G: γ=0.4",
    "G: γ=0.8",
]

SYNTHETIC_REPRESENTATIVE_SCENARIOS = [
    ("A: R=3, L=500, phi=0.5", "A (phi=0.5)"),
    ("B: L=1000, λ=0.4", "B (lambda=0.4)"),
    ("C: L=500, Δb=1.0", "C (Delta b=1.0)"),
    ("D: L_short=25", "D (L_short=25)"),
    ("F: ν=3", "F (nu=3)"),
    ("G: γ=0.8", "G (gamma=0.8)"),
]

SYNTHETIC_FULL_PART_1 = SYNTHETIC_SCENARIO_ORDER[:7]
SYNTHETIC_FULL_PART_2 = SYNTHETIC_SCENARIO_ORDER[7:]


def _get_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def dataset_sort_key(name: str) -> tuple[int, str]:
    try:
        return (DATASET_ORDER.index(name), name)
    except ValueError:
        return (len(DATASET_ORDER), name)


def label(dataset: str) -> str:
    return DATASET_LABELS.get(dataset, dataset)


def load_results(path: Path) -> pd.DataFrame:
    with open(path) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)
    if "E_series" not in df.columns and "E_r" in df.columns:
        warnings.warn(
            "Legacy real-data snapshot detected. Aliasing E_r to E_series for "
            "backward compatibility. Regenerate benchmark_results.json with "
            "benchmarks/run_real_data.py before treating the output as paper "
            "authoritative.",
            RuntimeWarning,
            stacklevel=2,
        )
        df["E_series"] = df["E_r"]
    if "T_eval" not in df.columns:
        warnings.warn(
            "Real-data snapshot is missing T_eval. This usually means the file "
            "predates the synchronized post-n_cal scoring horizon.",
            RuntimeWarning,
            stacklevel=2,
        )
    numeric = ["E_series", "coverage_90", "width_90", "resets", "ws_occ", "skipped", "T", "T_eval"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_synthetic_results(path: Path) -> pd.DataFrame:
    with open(path) as f:
        payload = json.load(f)
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    df = pd.DataFrame(rows)
    numeric = [
        "E_r",
        "coverage_80",
        "coverage_90",
        "width_80",
        "width_90",
        "pit_ks_p",
        "resets",
        "warm_start_occ",
        "wall_seconds",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _paper_datasets(df: pd.DataFrame) -> list[str]:
    return sorted(df["dataset"].dropna().unique(), key=dataset_sort_key)


def _synthetic_scenarios(df: pd.DataFrame) -> list[str]:
    present = set(df["scenario"].dropna().unique())
    ordered = [name for name in SYNTHETIC_SCENARIO_ORDER if name in present]
    extras = sorted(present.difference(SYNTHETIC_SCENARIO_ORDER))
    return ordered + extras


def dataset_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in _paper_datasets(df):
        sub = df[(df["dataset"] == dataset) & (df["method"] == "SCT")]
        if len(sub) == 0:
            continue
        rows.append(
            {
                "Dataset": label(dataset),
                "Series": int(sub["series_id"].nunique()),
                "Steps per series": int(round(sub["T"].median())),
                "Total steps": int(sub["T"].sum()),
                "Model": sub["model"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def real_data_runtime_table(df: pd.DataFrame, log_path: Path | None = None) -> pd.DataFrame:
    columns = [
        "Dataset",
        "Model",
        "Series",
        "Total seconds",
        "Seconds per series",
        "Series/hour",
    ]
    sct = df[df["method"] == "SCT"]
    runtime_by_key: dict[tuple[str, str], dict[str, float | int | str]] = {}
    for (dataset, model), sub in sct.groupby(["dataset", "model"]):
        runtime_by_key[(dataset, model)] = {
            "dataset": dataset,
            "model": model,
            "series": int(sub["series_id"].nunique()),
        }

    if log_path is not None and log_path.exists():
        start_re = re.compile(r"^(?P<dataset>[A-Za-z0-9_]+) / (?P<model>[^:]+): (?P<series>\d+) series$")
        done_re = re.compile(
            r"^\s*Completed (?P<dataset>[A-Za-z0-9_]+)/(?P<model>\S+) in (?P<seconds>[0-9.]+)s$"
        )
        with open(log_path) as f:
            for raw_line in f:
                line = raw_line.strip()
                match = start_re.match(line)
                if match is not None:
                    dataset = match.group("dataset")
                    model = match.group("model")
                    runtime_by_key.setdefault(
                        (dataset, model),
                        {"dataset": dataset, "model": model, "series": int(match.group("series"))},
                    )
                    runtime_by_key[(dataset, model)]["series"] = int(match.group("series"))
                    continue

                match = done_re.match(line)
                if match is None:
                    continue
                dataset = match.group("dataset")
                model = match.group("model")
                runtime_by_key.setdefault(
                    (dataset, model),
                    {"dataset": dataset, "model": model, "series": 0},
                )
                runtime_by_key[(dataset, model)]["total_seconds"] = float(match.group("seconds"))

    rows = []
    for (_, _), payload in sorted(
        runtime_by_key.items(),
        key=lambda item: (dataset_sort_key(item[0][0]), item[0][1]),
    ):
        total_seconds = payload.get("total_seconds")
        if total_seconds is None:
            continue
        series = int(payload["series"])
        series_per_hour = series / max(float(total_seconds), 1e-9) * 3600.0
        rows.append(
            {
                "Dataset": label(str(payload["dataset"])),
                "Model": payload["model"],
                "Series": series,
                "Total seconds": float(total_seconds),
                "Seconds per series": float(total_seconds) / max(series, 1),
                "Series/hour": series_per_hour,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def headline_real_data_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in _paper_datasets(df):
        for method in METHOD_ORDER:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if len(sub) == 0:
                continue
            rows.append(
                {
                    "Dataset": label(dataset),
                    "Method": method,
                    "Median E_series": sub["E_series"].median() if method == "SCT" else np.nan,
                    "Median Cov90": sub["coverage_90"].median(),
                    "Median W90": sub["width_90"].median(),
                    "Warm-start occupancy": sub["ws_occ"].median() if method == "SCT" else np.nan,
                }
            )
    return pd.DataFrame(rows)


def coverage_variability_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in _paper_datasets(df):
        for method in METHOD_ORDER:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if len(sub) == 0:
                continue
            cov = sub["coverage_90"].dropna()
            err = (cov - 0.90).abs()
            rows.append(
                {
                    "Dataset": label(dataset),
                    "Method": method,
                    "Median": cov.median(),
                    "IQR": cov.quantile(0.75) - cov.quantile(0.25),
                    "P10": cov.quantile(0.10),
                    "P90": cov.quantile(0.90),
                    "Failure rate": (err > 0.05).mean(),
                }
            )
    return pd.DataFrame(rows)


def _wilcoxon_safe(values: np.ndarray) -> float:
    try:
        return float(stats.wilcoxon(values).pvalue)
    except Exception:
        return float("nan")


def paired_wilcoxon_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in _paper_datasets(df):
        sct = df[(df["dataset"] == dataset) & (df["method"] == "SCT")].set_index("series_id")
        for comparator in ["ACI", "split-CP"]:
            comp = df[(df["dataset"] == dataset) & (df["method"] == comparator)].set_index("series_id")
            common = sct.index.intersection(comp.index)
            if len(common) == 0:
                continue

            cov_diff = sct.loc[common, "coverage_90"].values - comp.loc[common, "coverage_90"].values
            width_diff = sct.loc[common, "width_90"].values - comp.loc[common, "width_90"].values

            rows.append(
                {
                    "Dataset": label(dataset),
                    "Comparison": f"SCT vs {comparator}",
                    "N": int(len(common)),
                    "Cov diff": float(np.median(cov_diff)),
                    "Cov p": _wilcoxon_safe(cov_diff),
                    "Width diff": float(np.median(width_diff)),
                    "Width p": _wilcoxon_safe(width_diff),
                }
            )

            if comparator == "split-CP":
                mask = (
                    (sct.loc[common, "coverage_90"] - 0.90).abs() <= 0.05
                ) & (
                    (comp.loc[common, "coverage_90"] - 0.90).abs() <= 0.05
                )
                valid = common[mask]
                if len(valid) == 0:
                    continue
                width_diff_valid = sct.loc[valid, "width_90"].values - comp.loc[valid, "width_90"].values
                rows.append(
                    {
                        "Dataset": label(dataset),
                        "Comparison": "SCT vs split-CP†",
                        "N": int(len(valid)),
                        "Cov diff": np.nan,
                        "Cov p": np.nan,
                        "Width diff": float(np.median(width_diff_valid)),
                        "Width p": _wilcoxon_safe(width_diff_valid),
                    }
                )
    return pd.DataFrame(rows)


def synthetic_representative_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_name, display_name in SYNTHETIC_REPRESENTATIVE_SCENARIOS:
        sub = df[df["scenario"] == scenario_name].set_index("method")
        if not {"SCT", "ACI", "split-CP"}.issubset(sub.index):
            continue
        rows.append(
            {
                "Scenario": display_name,
                "E_r": sub.loc["SCT", "E_r"],
                "SCT Cov90": sub.loc["SCT", "coverage_90"],
                "ACI Cov90": sub.loc["ACI", "coverage_90"],
                "split-CP Cov90": sub.loc["split-CP", "coverage_90"],
                "SCT W90": sub.loc["SCT", "width_90"],
                "ACI W90": sub.loc["ACI", "width_90"],
                "split-CP W90": sub.loc["split-CP", "width_90"],
            }
        )
    return pd.DataFrame(rows)


def synthetic_representative_diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_name, display_name in SYNTHETIC_REPRESENTATIVE_SCENARIOS:
        sub = df[df["scenario"] == scenario_name].set_index("method")
        if not {"SCT", "ACI", "split-CP"}.issubset(sub.index):
            continue
        rows.append(
            {
                "Scenario": display_name,
                "PIT KS": sub.loc["SCT", "pit_ks_p"],
                "Resets": sub.loc["SCT", "resets"],
                "Warm-start": sub.loc["SCT", "warm_start_occ"],
                "SCT Time": sub.loc["SCT", "wall_seconds"],
                "ACI Time": sub.loc["ACI", "wall_seconds"],
                "split-CP Time": sub.loc["split-CP", "wall_seconds"],
            }
        )
    return pd.DataFrame(rows)


def synthetic_full_suite_table(df: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        sub = df[df["scenario"] == scenario]
        if len(sub) == 0:
            continue
        for method in METHOD_ORDER:
            method_row = sub[sub["method"] == method]
            if len(method_row) == 0:
                continue
            row = method_row.iloc[0]
            rows.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "E_r": row["E_r"],
                    "Cov90": row["coverage_90"],
                    "W90": row["width_90"],
                    "PIT KS": row["pit_ks_p"],
                    "Resets": row["resets"],
                    "Warm-start": row["warm_start_occ"],
                    "Time": row["wall_seconds"],
                }
            )
    return pd.DataFrame(rows)


def _apply_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def median_coverage_vs_width_figure(df: pd.DataFrame, out_dir: Path) -> None:
    plt = _get_plt()
    datasets = _paper_datasets(df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        _apply_style(ax)
        for method in METHOD_ORDER:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["coverage_90"].median(),
                sub["width_90"].median(),
                s=110,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLOURS[method],
                edgecolors="black",
                linewidth=0.5,
                label=method,
                zorder=3,
            )
        ax.axvline(0.90, color="gray", linestyle="--", alpha=0.5, label="Nominal 0.90")
        ax.set_xlabel("Median empirical coverage", fontsize=10)
        ax.set_ylabel("Median interval width", fontsize=10)
        ax.set_title(label(dataset), fontsize=11)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "median_coverage_vs_width.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "median_coverage_vs_width.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def eseries_distribution_figure(df: pd.DataFrame, out_dir: Path) -> None:
    plt = _get_plt()
    sct = df[df["method"] == "SCT"]
    datasets = _paper_datasets(sct)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        _apply_style(ax)
        sub = sct[sct["dataset"] == dataset]["E_series"].dropna()
        ax.hist(sub, bins=30, color=METHOD_COLOURS["SCT"], alpha=0.75, edgecolor="white")
        med = sub.median()
        ax.axvline(med, color="#c00000", linestyle="--", label=f"Median = {med:.4f}")
        ax.set_xlabel(r"$\mathcal{E}_{\mathrm{series}}$ (descriptive calibration score)", fontsize=10)
        ax.set_ylabel("Series count", fontsize=10)
        ax.set_title(f"{label(dataset)}  (n = {len(sub)})", fontsize=11)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "eseries_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "eseries_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


er_distribution_figure = eseries_distribution_figure


def coverage_distribution_figure(df: pd.DataFrame, out_dir: Path) -> None:
    plt = _get_plt()
    datasets = _paper_datasets(df)
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        _apply_style(ax)
        for method in METHOD_ORDER:
            sub = df[(df["dataset"] == dataset) & (df["method"] == method)]["coverage_90"].dropna()
            ax.hist(
                sub,
                bins=30,
                alpha=0.45,
                color=METHOD_COLOURS[method],
                edgecolor="white",
                label=method,
            )
        ax.axvline(0.90, color="black", linestyle="--", alpha=0.6, label="Nominal 0.90")
        ax.set_xlabel("Empirical coverage (90%)", fontsize=10)
        ax.set_ylabel("Series count", fontsize=10)
        ax.set_title(label(dataset), fontsize=11)
        ax.legend(fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / "coverage_distribution.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "coverage_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_paper_artifacts(
    df: pd.DataFrame,
    out_root: Path,
    synthetic_df: pd.DataFrame | None = None,
    real_data_log_path: Path | None = None,
) -> dict[str, list[Path]]:
    table_dir = out_root / "tables"
    fig_dir = out_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset_summary = dataset_summary_table(df)
    headline_real_data = headline_real_data_table(df)
    coverage_variability = coverage_variability_table(df)
    paired_wilcoxon = paired_wilcoxon_table(df)
    real_data_runtime = real_data_runtime_table(df, log_path=real_data_log_path)

    dataset_summary.to_csv(table_dir / "dataset_summary.csv", index=False)
    headline_real_data.to_csv(table_dir / "headline_real_data.csv", index=False)
    coverage_variability.to_csv(table_dir / "coverage_variability.csv", index=False)
    paired_wilcoxon.to_csv(table_dir / "paired_wilcoxon.csv", index=False)
    real_data_runtime.to_csv(table_dir / "real_data_runtime_summary.csv", index=False)

    table_files = [table_dir / name for name in PAPER_TABLE_FILES]

    if synthetic_df is not None:
        synthetic_representative_summary = synthetic_representative_summary_table(synthetic_df)
        synthetic_representative_diagnostics = synthetic_representative_diagnostics_table(synthetic_df)
        synthetic_full_part_1 = synthetic_full_suite_table(synthetic_df, SYNTHETIC_FULL_PART_1)
        synthetic_full_part_2 = synthetic_full_suite_table(synthetic_df, SYNTHETIC_FULL_PART_2)

        synthetic_representative_summary.to_csv(
            table_dir / "synthetic_representative_summary.csv", index=False
        )
        synthetic_representative_diagnostics.to_csv(
            table_dir / "synthetic_representative_diagnostics.csv", index=False
        )
        synthetic_full_part_1.to_csv(
            table_dir / "synthetic_full_suite_part1.csv", index=False
        )
        synthetic_full_part_2.to_csv(
            table_dir / "synthetic_full_suite_part2.csv", index=False
        )
        table_files.extend(table_dir / name for name in SYNTHETIC_TABLE_FILES)

    median_coverage_vs_width_figure(df, fig_dir)
    eseries_distribution_figure(df, fig_dir)
    coverage_distribution_figure(df, fig_dir)

    return {
        "tables": table_files,
        "figures": [fig_dir / name for name in PAPER_FIGURE_FILES],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/benchmark_results.json")
    parser.add_argument("--synthetic-results", default="data/results/synthetic_results.json")
    parser.add_argument("--real-data-log", default=None)
    parser.add_argument("--output", default="analysis/output/paper")
    args = parser.parse_args()

    results_path = Path(args.results)
    synthetic_results_path = Path(args.synthetic_results)
    real_data_log_path = Path(args.real_data_log) if args.real_data_log else None
    out_root = Path(args.output)
    df = load_results(results_path)
    synthetic_df = None
    if synthetic_results_path.exists():
        synthetic_df = load_synthetic_results(synthetic_results_path)
    outputs = write_paper_artifacts(
        df,
        out_root,
        synthetic_df=synthetic_df,
        real_data_log_path=real_data_log_path,
    )

    print(f"Loaded {len(df)} rows from {results_path}")
    if synthetic_df is not None:
        print(f"Loaded {len(synthetic_df)} synthetic rows from {synthetic_results_path}")
    print(f"Wrote paper artifacts to {out_root}")
    print(f"Tables: {[path.name for path in outputs['tables']]}")
    print(f"Figures: {[path.name for path in outputs['figures']]}")


if __name__ == "__main__":
    main()
