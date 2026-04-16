import numpy as np
from numpy.testing import assert_allclose


def test_regimewise_calibration_error_is_length_weighted():
    from benchmarks.run_synthetic import regimewise_calibration_error
    from tsconformal.metrics import gridwise_calibration_error

    y = np.zeros(4)
    calibrated_quantiles = np.array([
        [-1.0, 0.1, 0.1],
        [-1.0, 0.1, 0.1],
        [-1.0, -1.0, 0.1],
        [-1.0, -1.0, 0.1],
    ])
    regime_ids = np.array([0, 0, 1, 1])

    whole_stream = gridwise_calibration_error(y, calibrated_quantiles, J_eval=3)
    regimewise = regimewise_calibration_error(
        y,
        calibrated_quantiles,
        regime_ids,
        J_eval=3,
    )

    e0 = gridwise_calibration_error(
        y[regime_ids == 0],
        calibrated_quantiles[regime_ids == 0],
        J_eval=3,
    )
    e1 = gridwise_calibration_error(
        y[regime_ids == 1],
        calibrated_quantiles[regime_ids == 1],
        J_eval=3,
    )
    expected = np.average([e0, e1], weights=[2, 2])

    assert_allclose(regimewise, expected)
    assert regimewise > whole_stream


def test_run_sct_uses_regimewise_calibration_error(monkeypatch):
    import benchmarks.run_synthetic as run_synthetic
    from benchmarks.synthetic.dgp import SyntheticStream

    sentinel = 0.123456

    def fake_regimewise(y, calibrated_quantiles, regime_ids, J_eval=99):
        assert len(y) == len(regime_ids)
        assert calibrated_quantiles.shape[0] == len(y)
        return sentinel

    monkeypatch.setattr(
        run_synthetic,
        "regimewise_calibration_error",
        fake_regimewise,
    )

    stream = SyntheticStream(
        y=np.array([0.0, 0.1, -0.2, 0.3]),
        forecast_mu=np.zeros(4),
        forecast_sigma=np.ones(4),
        regime_ids=np.array([0, 0, 1, 1]),
        changepoints=[2],
        family="test",
        params={},
    )

    result = run_synthetic.run_sct(stream)

    assert_allclose(result.E_r, sentinel)
