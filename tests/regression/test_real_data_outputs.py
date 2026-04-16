import json
import os
import warnings

import numpy as np
import pytest


def _toy_forecasts():
    forecasts = []
    for t in range(12):
        if t < 4:
            quantiles = [10.0, 10.0, 10.0]
        else:
            quantiles = [0.0, 0.0, 0.0]
        forecasts.append(
            {
                "probabilities": [0.05, 0.50, 0.95],
                "quantiles": quantiles,
                "y_true": 0.0,
            }
        )
    return forecasts


def test_real_data_methods_share_post_ncal_horizon():
    from benchmarks.run_real_data import (
        run_aci_on_cached,
        run_sct_on_cached,
        run_split_cp_on_cached,
    )

    forecasts = _toy_forecasts()

    sct = run_sct_on_cached(forecasts, eval_start=4)
    aci = run_aci_on_cached(forecasts, eval_start=4)
    split_cp = run_split_cp_on_cached(forecasts, n_cal=4)

    for row in [sct, aci, split_cp]:
        assert row["T"] == 12
        assert row["T_eval"] == 8
        assert row["coverage_90"] == 1.0

    assert "E_series" in sct
    assert "E_r" not in sct
    assert np.isfinite(sct["E_series"])
    assert np.isnan(aci["E_series"])
    assert np.isnan(split_cp["E_series"])


def test_paper_artifact_loader_aliases_old_er_schema(tmp_path):
    os.environ["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")
    from analysis.generate_paper_artifacts import load_results, headline_real_data_table

    path = tmp_path / "results.json"
    with open(path, "w") as f:
        json.dump(
            [
                {
                    "dataset": "fred_md",
                    "model": "chronos2",
                    "series_id": "T1",
                    "method": "SCT",
                    "E_r": 0.125,
                    "coverage_90": 0.9,
                    "width_90": 1.0,
                    "ws_occ": 1.0,
                    "T": 100,
                }
            ],
            f,
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_results(path)
    table = headline_real_data_table(df)

    assert "E_series" in df.columns
    assert df.loc[0, "E_series"] == 0.125
    assert "Median E_series" in table.columns
    messages = [str(item.message) for item in caught]
    assert any("Aliasing E_r to E_series" in message for message in messages)
    assert any("missing T_eval" in message for message in messages)


def test_cache_cli_exposes_only_implemented_models():
    from benchmarks.cache_fm_forecasts import SUPPORTED_MODELS, build_parser

    parser = build_parser()
    model_action = next(action for action in parser._actions if action.dest == "model")

    assert sorted(SUPPORTED_MODELS) == ["chronos2"]
    assert sorted(model_action.choices) == ["chronos2"]

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--dataset", "fred_md", "--model", "timesfm"])
    assert excinfo.value.code == 2


def test_real_data_parser_accepts_explicit_pit_seed():
    from benchmarks.run_real_data import build_parser

    parser = build_parser()
    args = parser.parse_args(["--pit-seed", "123"])

    assert args.pit_seed == 123


def test_run_sct_on_cached_accepts_explicit_pit_seed():
    from benchmarks.run_real_data import run_sct_on_cached

    forecasts = _toy_forecasts()

    left = run_sct_on_cached(forecasts, eval_start=4, pit_seed=123)
    right = run_sct_on_cached(forecasts, eval_start=4, pit_seed=123)

    assert left == right


def test_unimplemented_cache_helpers_fail_explicitly(tmp_path):
    from benchmarks.cache_fm_forecasts import cache_moirai, cache_timesfm

    for helper, label in [(cache_timesfm, "TimesFM"), (cache_moirai, "Moirai")]:
        with pytest.raises(NotImplementedError, match=label):
            helper(df=None, output_dir=tmp_path)
