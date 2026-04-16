import numpy as np
from numpy.testing import assert_allclose


def test_point_mass_quantile_grid_is_valid_discrete_forecast():
    from tsconformal import QuantileGridCDFAdapter, RandomizedPIT
    from tsconformal.forecast import validate_forecast_cdf

    adapter = QuantileGridCDFAdapter(
        probabilities=[0.1, 0.5, 0.9],
        quantiles=[264.0, 264.0, 264.0],
    )

    validate_forecast_cdf(adapter)

    assert adapter.is_discrete
    assert adapter.cdf_left(264.0) == 0.0
    assert adapter.cdf(263.9) == 0.0
    assert adapter.cdf(264.0) == 1.0
    assert adapter.cdf(264.1) == 1.0
    assert_allclose(adapter.ppf(np.array([0.1, 0.5, 0.9])), 264.0)

    rng = np.random.default_rng(42)
    pits = np.array([RandomizedPIT.pit(adapter, 264.0, rng) for _ in range(2000)])
    assert np.all((pits >= 0.0) & (pits <= 1.0))
    assert 0.47 < pits.mean() < 0.53


def test_run_sct_on_cached_scores_point_mass_rows():
    from benchmarks.run_real_data import run_sct_on_cached

    forecasts = [
        {
            "probabilities": [0.05, 0.50, 0.95],
            "quantiles": [264.0, 264.0, 264.0],
            "y_true": 264.0,
        }
        for _ in range(4)
    ]

    result = run_sct_on_cached(forecasts, alpha=0.10, eval_start=0)

    assert result["status"] == "ok"
    assert result["skipped"] == 0
    assert result["T"] == 4
    assert result["T_eval"] == 4
    assert np.isfinite(result["E_series"])
    assert result["coverage_90"] == 1.0
    assert result["width_90"] == 0.0
