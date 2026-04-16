import json
import os
from pathlib import Path

import pandas as pd


def _tiny_results():
    return [
        {
            "dataset": "electricity_cached",
            "model": "chronos2",
            "series_id": "S1",
            "method": "SCT",
            "E_series": 0.01,
            "coverage_90": 0.89,
            "width_90": 10.0,
            "ws_occ": 0.4,
            "T": 100,
            "T_eval": 64,
        },
        {
            "dataset": "electricity_cached",
            "model": "chronos2",
            "series_id": "S1",
            "method": "ACI",
            "E_series": None,
            "coverage_90": 0.91,
            "width_90": 11.0,
            "T": 100,
            "T_eval": 64,
        },
        {
            "dataset": "electricity_cached",
            "model": "chronos2",
            "series_id": "S1",
            "method": "split-CP",
            "E_series": None,
            "coverage_90": 0.90,
            "width_90": 12.0,
            "T": 100,
            "T_eval": 64,
        },
        {
            "dataset": "electricity_cached",
            "model": "chronos2",
            "series_id": "S2",
            "method": "SCT",
            "E_series": 0.02,
            "coverage_90": 0.88,
            "width_90": 9.0,
            "ws_occ": 0.5,
            "T": 120,
            "T_eval": 84,
        },
        {
            "dataset": "electricity_cached",
            "model": "chronos2",
            "series_id": "S2",
            "method": "ACI",
            "E_series": None,
            "coverage_90": 0.92,
            "width_90": 10.0,
            "T": 120,
            "T_eval": 84,
        },
        {
            "dataset": "electricity_cached",
            "model": "chronos2",
            "series_id": "S2",
            "method": "split-CP",
            "E_series": None,
            "coverage_90": 0.91,
            "width_90": 11.0,
            "T": 120,
            "T_eval": 84,
        },
    ]


def _tiny_synthetic_snapshot():
    return {
        "script": "benchmarks/run_synthetic.py",
        "n_replicates": 2,
        "seed_base": 123,
        "n_rows": 6,
        "results": [
            {
                "scenario": "A: R=3, L=500, phi=0.5",
                "method": "SCT",
                "E_r": 0.04,
                "coverage_90": 0.86,
                "coverage_80": 0.76,
                "width_90": 3.1,
                "width_80": 2.4,
                "pit_ks_p": 0.02,
                "resets": 15,
                "warm_start_occ": 0.42,
                "wall_seconds": 27.7,
            },
            {
                "scenario": "A: R=3, L=500, phi=0.5",
                "method": "ACI",
                "E_r": None,
                "coverage_90": 0.90,
                "coverage_80": 0.80,
                "width_90": 3.4,
                "width_80": 2.7,
                "pit_ks_p": None,
                "resets": 0,
                "warm_start_occ": None,
                "wall_seconds": 0.7,
            },
            {
                "scenario": "A: R=3, L=500, phi=0.5",
                "method": "split-CP",
                "E_r": None,
                "coverage_90": 0.88,
                "coverage_80": 0.78,
                "width_90": 3.23,
                "width_80": 2.5,
                "pit_ks_p": None,
                "resets": 0,
                "warm_start_occ": None,
                "wall_seconds": 0.0,
            },
            {
                "scenario": "D: L_short=25",
                "method": "SCT",
                "E_r": 0.18,
                "coverage_90": 0.84,
                "coverage_80": 0.75,
                "width_90": 3.7,
                "width_80": 3.0,
                "pit_ks_p": 0.00005,
                "resets": 30,
                "warm_start_occ": 0.42,
                "wall_seconds": 54.4,
            },
            {
                "scenario": "D: L_short=25",
                "method": "ACI",
                "E_r": None,
                "coverage_90": 0.90,
                "coverage_80": 0.80,
                "width_90": 3.67,
                "width_80": 2.87,
                "pit_ks_p": None,
                "resets": 0,
                "warm_start_occ": None,
                "wall_seconds": 1.5,
            },
            {
                "scenario": "D: L_short=25",
                "method": "split-CP",
                "E_r": None,
                "coverage_90": 0.90,
                "coverage_80": 0.80,
                "width_90": 3.66,
                "width_80": 2.84,
                "pit_ks_p": None,
                "resets": 0,
                "warm_start_occ": None,
                "wall_seconds": 0.0,
            },
        ],
    }


def test_generate_paper_artifacts_matches_paper_figure_names(tmp_path):
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")

    import analysis.generate_paper_artifacts as paper_artifacts
    from analysis.generate_paper_artifacts import (
        PAPER_FIGURE_FILES,
        PAPER_TABLE_FILES,
        load_results,
        write_paper_artifacts,
    )

    results_path = tmp_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(_tiny_results(), f)

    def _write_stub(name):
        def _inner(df, out_dir):
            stem = Path(name).stem
            (out_dir / f"{stem}.pdf").write_text("stub")
            (out_dir / name).write_text("stub")
        return _inner

    paper_artifacts.median_coverage_vs_width_figure = _write_stub(
        "median_coverage_vs_width.png"
    )
    paper_artifacts.eseries_distribution_figure = _write_stub(
        "eseries_distribution.png"
    )
    paper_artifacts.coverage_distribution_figure = _write_stub(
        "coverage_distribution.png"
    )

    df = load_results(results_path)
    out_root = tmp_path / "paper"
    outputs = write_paper_artifacts(df, out_root)

    for path in outputs["tables"] + outputs["figures"]:
        assert path.exists(), path

    assert [path.name for path in outputs["tables"]] == list(PAPER_TABLE_FILES)
    assert [path.name for path in outputs["figures"]] == list(PAPER_FIGURE_FILES)
    for figure_name in PAPER_FIGURE_FILES:
        pdf_name = f"{Path(figure_name).stem}.pdf"
        assert (out_root / "figures" / pdf_name).exists()


def test_generate_paper_artifacts_includes_synthetic_tables_when_snapshot_present(tmp_path):
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")

    import analysis.generate_paper_artifacts as paper_artifacts
    from analysis.generate_paper_artifacts import (
        SYNTHETIC_TABLE_FILES,
        load_results,
        load_synthetic_results,
        write_paper_artifacts,
    )

    results_path = tmp_path / "results.json"
    synthetic_path = tmp_path / "synthetic.json"
    with open(results_path, "w") as f:
        json.dump(_tiny_results(), f)
    with open(synthetic_path, "w") as f:
        json.dump(_tiny_synthetic_snapshot(), f)

    def _write_stub(name):
        def _inner(df, out_dir):
            stem = Path(name).stem
            (out_dir / f"{stem}.pdf").write_text("stub")
            (out_dir / name).write_text("stub")
        return _inner

    paper_artifacts.median_coverage_vs_width_figure = _write_stub(
        "median_coverage_vs_width.png"
    )
    paper_artifacts.eseries_distribution_figure = _write_stub(
        "eseries_distribution.png"
    )
    paper_artifacts.coverage_distribution_figure = _write_stub(
        "coverage_distribution.png"
    )

    df = load_results(results_path)
    synthetic_df = load_synthetic_results(synthetic_path)
    out_root = tmp_path / "paper"
    outputs = write_paper_artifacts(df, out_root, synthetic_df=synthetic_df)

    table_names = [path.name for path in outputs["tables"]]
    for expected_name in SYNTHETIC_TABLE_FILES:
        assert expected_name in table_names

    rep = pd.read_csv(out_root / "tables" / "synthetic_representative_summary.csv")
    assert list(rep["Scenario"]) == ["A (phi=0.5)", "D (L_short=25)"]
    assert rep.loc[0, "SCT Cov90"] == 0.86
    assert rep.loc[1, "split-CP W90"] == 3.66

    full_part_1 = pd.read_csv(out_root / "tables" / "synthetic_full_suite_part1.csv")
    full_part_2 = pd.read_csv(out_root / "tables" / "synthetic_full_suite_part2.csv")
    assert set(full_part_1["Scenario"]) == {"A: R=3, L=500, phi=0.5"}
    assert set(full_part_2["Scenario"]) == {"D: L_short=25"}


def test_generate_paper_artifacts_includes_real_data_runtime_table_when_log_present(tmp_path):
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")

    import analysis.generate_paper_artifacts as paper_artifacts
    from analysis.generate_paper_artifacts import load_results, write_paper_artifacts

    results_path = tmp_path / "results.json"
    with open(results_path, "w") as f:
        json.dump(_tiny_results(), f)

    log_path = tmp_path / "postcache_reproduction.log"
    log_path.write_text(
        "\n".join(
            [
                "electricity_cached / chronos2: 2 series",
                "  Completed electricity_cached/chronos2 in 12.5s",
            ]
        )
        + "\n"
    )

    def _write_stub(name):
        def _inner(df, out_dir):
            stem = Path(name).stem
            (out_dir / f"{stem}.pdf").write_text("stub")
            (out_dir / name).write_text("stub")
        return _inner

    paper_artifacts.median_coverage_vs_width_figure = _write_stub(
        "median_coverage_vs_width.png"
    )
    paper_artifacts.eseries_distribution_figure = _write_stub(
        "eseries_distribution.png"
    )
    paper_artifacts.coverage_distribution_figure = _write_stub(
        "coverage_distribution.png"
    )

    df = load_results(results_path)
    out_root = tmp_path / "paper"
    outputs = write_paper_artifacts(df, out_root, real_data_log_path=log_path)

    table_names = [path.name for path in outputs["tables"]]
    assert "real_data_runtime_summary.csv" in table_names

    runtime = pd.read_csv(out_root / "tables" / "real_data_runtime_summary.csv")
    assert list(runtime.columns) == [
        "Dataset",
        "Model",
        "Series",
        "Total seconds",
        "Seconds per series",
        "Series/hour",
    ]
    assert runtime.loc[0, "Dataset"] == "Electricity Hourly"
    assert runtime.loc[0, "Model"] == "chronos2"
    assert runtime.loc[0, "Series"] == 2
    assert runtime.loc[0, "Total seconds"] == 12.5
