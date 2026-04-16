import warnings

import numpy as np


def _toy_stream(length: int = 12):
    from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

    return [(GaussianForecastCDF(0.0, 1.0), 0.0) for _ in range(length)]


def _benchmark_style_occupancy(stream, config):
    from tsconformal.calibrators import SegmentedTransportCalibrator
    from tsconformal.detectors import CUSUMNormDetector, PageHinkleyDetector
    from tsconformal.diagnostics import warm_start_occupancy
    from tsconformal.forecast import InvalidForecastCDFError

    if config.detector_type == "CUSUMNorm":
        det = CUSUMNormDetector(
            kappa=config.kappa,
            threshold=config.threshold,
        )
    elif config.detector_type == "PageHinkley":
        det = PageHinkleyDetector(
            delta=config.extra.get("delta", 0.01),
            threshold=config.threshold,
        )
    else:
        raise AssertionError(f"unsupported detector_type in test: {config.detector_type}")

    cal = SegmentedTransportCalibrator(
        grid_size=config.extra.get("grid_size", 49),
        rho=config.extra.get("rho", 0.99),
        n_eff_min=config.extra.get("n_eff_min", 50.0),
        step_schedule=lambda n: min(0.20, 1.0 / max(n ** 0.5, 1e-8)),
        detector=det,
        cooldown=config.extra.get("cooldown", 168),
        confirm=config.extra.get("confirm", 3),
    )

    pit_rng = np.random.default_rng(20260306)
    warmup_flags = []
    warm_start_weights = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for base_cdf, y_t in stream:
            try:
                cal.predict_cdf(base_cdf)
                warmup_flags.append(cal.in_warmup)
                warm_start_weights.append(cal.warm_start_weight)
                cal.update(y_t, base_cdf, rng=pit_rng)
            except InvalidForecastCDFError:
                continue

    flags = np.asarray(warmup_flags, dtype=bool)
    weights = np.asarray(warm_start_weights, dtype=np.float64)
    return warm_start_occupancy(flags, weights), float(np.mean(flags))


def test_sensitivity_report_matches_benchmark_warm_start_occupancy():
    from tsconformal.diagnostics import DetectorConfig, sensitivity_report

    stream = _toy_stream()
    config = DetectorConfig(
        threshold=1e6,
        extra={"n_eff_min": 3.0, "cooldown": 100000, "confirm": 3},
    )

    expected, fallback_only = _benchmark_style_occupancy(stream, config)
    report = sensitivity_report([config], stream)

    assert expected > fallback_only
    assert len(report.results) == 1
    assert np.isclose(report.results[0].warm_start_occupancy, expected)


def test_sensitivity_report_occupancy_skips_invalid_forecasts():
    from tsconformal.diagnostics import DetectorConfig, sensitivity_report

    class InvalidCDF:
        is_discrete = False

        def cdf(self, y):
            return 0.5

        def ppf(self, u):
            return np.nan

    valid_stream = _toy_stream(length=6)
    mixed_stream = [(InvalidCDF(), 0.0), *valid_stream]
    config = DetectorConfig(
        threshold=1e6,
        extra={"n_eff_min": 3.0, "cooldown": 100000, "confirm": 3},
    )

    expected, _ = _benchmark_style_occupancy(valid_stream, config)
    report = sensitivity_report([config], mixed_stream)

    assert len(report.results) == 1
    assert np.isclose(report.results[0].warm_start_occupancy, expected)


def test_sensitivity_report_mean_segment_length_uses_valid_steps_only():
    from tsconformal.diagnostics import DetectorConfig, sensitivity_report

    class InvalidCDF:
        is_discrete = False

        def cdf(self, y):
            return 0.5

        def ppf(self, u):
            return np.nan

    valid_stream = _toy_stream(length=2)
    mixed_stream = [*valid_stream, *((InvalidCDF(), 0.0),) * 8]
    report = sensitivity_report([DetectorConfig(threshold=1e6)], mixed_stream)

    assert len(report.results) == 1
    assert np.isclose(report.results[0].mean_segment_length, 2.0)


def test_sensitivity_report_accepts_explicit_pit_seed():
    from tsconformal.diagnostics import DetectorConfig, sensitivity_report

    stream = _toy_stream(length=6)
    config = DetectorConfig(
        threshold=1e6,
        extra={"n_eff_min": 3.0, "cooldown": 100000, "confirm": 3},
    )

    default_report = sensitivity_report([config], stream)
    seeded_report = sensitivity_report([config], stream, pit_seed=123)

    assert len(default_report.results) == 1
    assert len(seeded_report.results) == 1
    assert np.isclose(
        seeded_report.results[0].warm_start_occupancy,
        default_report.results[0].warm_start_occupancy,
    )
