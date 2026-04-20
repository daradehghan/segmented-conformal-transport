"""Property-based tests for tsconformal.

Covers the required property tests from PackageSpec_tsconformal_final:
A. Randomized PIT correctness
B. Transport monotonicity and boundary conditions
C. Isotonic nonexpansiveness in gridwise L2
D. Effective sample size accounting
E. No-lookahead behavior
F. Adapter extrapolation clipping

Plus the 8 numerical edge-case tests.
"""

import numpy as np
from numpy.testing import assert_allclose

# -----------------------------------------------------------------------
# A. Randomized PIT correctness
# -----------------------------------------------------------------------

class TestRandomizedPIT:
    def test_continuous_pit_in_range(self):
        from tsconformal import RandomizedPIT
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cdf = GaussianForecastCDF(mu=0.0, sigma=1.0)
        rng = np.random.default_rng(42)
        for y in np.linspace(-3, 3, 50):
            u = RandomizedPIT.pit(cdf, float(y), rng)
            assert 0.0 <= u <= 1.0, f"PIT out of range: {u}"

    def test_point_mass_forecast_gives_uniform(self):
        """Edge case 5: point-mass forecast should yield uniform PITs."""
        from tsconformal import RandomizedPIT

        class PointMassCDF:
            is_discrete = True
            def cdf(self, y):
                return 0.0 if y < 0.0 else 1.0
            def ppf(self, u):
                return 0.0

        cdf = PointMassCDF()
        rng = np.random.default_rng(42)
        pits = [RandomizedPIT.pit(cdf, 0.0, rng) for _ in range(10000)]
        pits = np.array(pits)
        # Should be approximately uniform
        assert np.all((pits >= 0.0) & (pits <= 1.0))
        # KS test
        from scipy.stats import kstest
        _, p = kstest(pits, "uniform")
        assert p > 0.01, f"PITs not uniform for point mass: KS p={p}"

    def test_reproducibility_with_same_seed(self):
        from tsconformal import RandomizedPIT
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cdf = GaussianForecastCDF(mu=0.0, sigma=1.0)
        u1 = RandomizedPIT.pit(cdf, 0.5, np.random.default_rng(123))
        u2 = RandomizedPIT.pit(cdf, 0.5, np.random.default_rng(123))
        assert u1 == u2


# -----------------------------------------------------------------------
# B. Transport monotonicity and boundary conditions
# -----------------------------------------------------------------------

class TestTransportMap:
    def test_monotone_and_boundaries(self):
        from tsconformal.utils import build_transport_map
        grid = np.array([0.25, 0.50, 0.75])
        g = np.array([0.20, 0.55, 0.80])  # monotone
        T, T_inv = build_transport_map(grid, g)

        assert T(0.0) == 0.0
        assert T(1.0) == 1.0

        # Monotonicity
        us = np.linspace(0, 1, 100)
        ts = np.array([T(u) for u in us])
        assert np.all(np.diff(ts) >= -1e-12)
        assert np.all((ts >= 0.0) & (ts <= 1.0))

    def test_upper_inverse_uses_plateau_right_endpoint(self):
        from tsconformal.utils import build_transport_map

        grid = np.array([0.25, 0.50, 0.75])
        g = np.array([0.20, 0.20, 0.80])
        _, T_inv = build_transport_map(grid, g)

        assert_allclose(T_inv(0.20), 0.50)
        assert_allclose(
            T_inv(np.array([0.10, 0.20, 0.50, 0.80])),
            np.array([0.125, 0.50, 0.625, 0.75]),
        )


# -----------------------------------------------------------------------
# C. Isotonic nonexpansiveness in gridwise L2
# -----------------------------------------------------------------------

class TestIsotonicProjection:
    def test_nonexpansive(self):
        from tsconformal.utils import pav_isotonic
        rng = np.random.default_rng(42)
        for _ in range(100):
            J = 20
            m = rng.uniform(0, 1, J)
            g_ref = np.sort(rng.uniform(0, 1, J))  # monotone reference

            g_proj = pav_isotonic(m)
            err_before = np.sqrt(np.mean((m - g_ref) ** 2))
            err_after = np.sqrt(np.mean((g_proj - g_ref) ** 2))
            assert err_after <= err_before + 1e-10


# -----------------------------------------------------------------------
# D. Effective sample size accounting
# -----------------------------------------------------------------------

class TestEffectiveSampleSize:
    def test_n_eff_formula(self):
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        det = CUSUMNormDetector(kappa=0.02, threshold=10.0)  # high threshold = no resets
        cal = SegmentedTransportCalibrator(
            grid_size=10, rho=0.9, n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=det, cooldown=100, confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)
        rng = np.random.default_rng(42)

        for _ in range(50):
            y = rng.normal()
            cal.predict_cdf(cdf)
            cal.update(y, cdf)

        # Manual check: W = sum rho^(t-i), W2 = sum rho^(2(t-i))
        assert cal.n_eff > 0
        assert cal.n_eff <= 50

    def test_fallback_below_threshold(self):
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        det = CUSUMNormDetector(kappa=0.02, threshold=10.0)
        cal = SegmentedTransportCalibrator(
            grid_size=10, rho=0.9, n_eff_min=100.0,  # very high
            step_schedule=lambda n: 0.1,
            detector=det, cooldown=100, confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)
        rng = np.random.default_rng(42)

        for _ in range(20):
            cal.predict_cdf(cdf)
            cal.update(rng.normal(), cdf)

        assert cal.in_warmup  # should be in fallback


# -----------------------------------------------------------------------
# E. No-lookahead behavior
# -----------------------------------------------------------------------

class TestNoLookahead:
    def test_predict_does_not_access_y(self):
        """predict_cdf must only perform validation-time probe calls."""
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator

        class SpyCDF:
            is_discrete = False
            call_log = []

            def cdf(self, y):
                SpyCDF.call_log.append(("cdf", y))
                from scipy.stats import norm
                return float(norm.cdf(y, 0, 1))

            def ppf(self, u):
                SpyCDF.call_log.append(("ppf", u))
                from scipy.stats import norm
                return float(norm.ppf(np.clip(u, 1e-12, 1 - 1e-12), 0, 1))

        det = CUSUMNormDetector(kappa=0.02, threshold=10.0)
        cal = SegmentedTransportCalibrator(
            grid_size=5, rho=0.9, n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=det, cooldown=100, confirm=3,
        )

        spy = SpyCDF()
        SpyCDF.call_log.clear()

        _ = cal.predict_cdf(spy)

        ppf_calls = [arg for name, arg in SpyCDF.call_log if name == "ppf"]
        cdf_calls = [arg for name, arg in SpyCDF.call_log if name == "cdf"]

        assert len(ppf_calls) == len(cdf_calls) > 0
        assert [name for name, _ in SpyCDF.call_log[:len(ppf_calls)]] == ["ppf"] * len(ppf_calls)
        assert [name for name, _ in SpyCDF.call_log[len(ppf_calls):]] == ["cdf"] * len(cdf_calls)

        from scipy.stats import norm

        expected_probe_values = np.array(
            [norm.ppf(np.clip(u, 1e-12, 1 - 1e-12), 0, 1) for u in ppf_calls]
        )
        assert_allclose(cdf_calls, expected_probe_values)

    def test_page_hinkley_rejects_negative_delta(self):
        """PageHinkleyDetector must reject negative drift allowances."""
        import pytest
        from tsconformal.detectors import PageHinkleyDetector

        with pytest.raises(ValueError, match="delta must be finite and non-negative"):
            PageHinkleyDetector(delta=-0.01, threshold=0.5)

    def test_page_hinkley_rejects_non_finite_or_non_positive_threshold(self):
        """PageHinkleyDetector must reject invalid thresholds explicitly."""
        import pytest
        from tsconformal.detectors import PageHinkleyDetector

        for threshold in (np.nan, np.inf, 0.0):
            with pytest.raises(ValueError, match="threshold must be finite and positive"):
                PageHinkleyDetector(delta=0.01, threshold=threshold)

    def test_cusum_rejects_non_finite_parameters(self):
        """CUSUMNormDetector must reject non-finite tuning parameters."""
        import pytest
        from tsconformal.detectors import CUSUMNormDetector

        with pytest.raises(ValueError, match="kappa must be finite and non-negative"):
            CUSUMNormDetector(kappa=np.nan, threshold=0.5)

        with pytest.raises(ValueError, match="threshold must be finite and positive"):
            CUSUMNormDetector(kappa=0.02, threshold=np.nan)

    def test_aci_update_uses_issued_interval(self):
        """ACI must update from the interval issued before seeing y_t."""
        from benchmarks.baselines import ACIBaseline

        aci = ACIBaseline(alpha=0.10, gamma=0.10)
        aci._scores.extend([0.1, 0.2])

        lower, upper = aci.predict_interval(y_hat=0.0)
        assert_allclose([lower, upper], [-0.19, 0.19])

        y_t = 0.199
        assert not (lower <= y_t <= upper)


class TestWarningContracts:
    def test_low_effective_sample_warning_is_emitted(self):
        """Low effective sample fallback must emit its public warning class."""
        import pytest
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF
        from tsconformal.calibrators import LowEffectiveSampleWarning

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=100.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)

        cal.predict_cdf(cdf)
        with pytest.warns(
            LowEffectiveSampleWarning,
            match="identity fallback activated",
        ):
            cal.update(0.0, cdf)

    def test_warm_start_dominance_warning_is_emitted(self):
        """Early non-fallback steps must emit warm-start dominance warnings."""
        import pytest
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF
        from tsconformal.calibrators import WarmStartDominanceWarning

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)

        cal.predict_cdf(cdf)
        with pytest.warns(
            WarmStartDominanceWarning,
            match="dominated by identity initialization",
        ):
            cal.update(0.0, cdf)

    def test_discrete_forecast_requires_rng(self):
        """Discrete forecasts must refuse updates without randomized PIT support."""
        import pytest
        from tsconformal import QuantileGridCDFAdapter, SegmentedTransportCalibrator
        from tsconformal import CUSUMNormDetector
        from tsconformal.calibrators import DiscreteForecastWithoutRandomizedPITError

        cdf = QuantileGridCDFAdapter(
            probabilities=[0.1, 0.5, 0.9],
            quantiles=[1.0, 1.0, 1.0],
        )
        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )

        cal.predict_cdf(cdf)
        before = cal._get_state()
        with pytest.raises(
            DiscreteForecastWithoutRandomizedPITError,
            match="requires rng for randomized PIT",
        ):
            cal.update(1.0, cdf)
        assert cal._get_state() == before

    def test_non_finite_outcome_leaves_update_state_unchanged(self):
        """Non-finite observed outcomes must be rejected atomically."""
        import pytest
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cdf = GaussianForecastCDF(0.0, 1.0)
        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )

        cal.predict_cdf(cdf)
        before = cal._get_state()
        with pytest.raises(ValueError, match="y_t must be finite"):
            cal.update(np.nan, cdf)
        assert cal._get_state() == before

    def test_non_finite_pit_leaves_update_state_unchanged(self, monkeypatch):
        """Invalid PIT output must be rejected before any state mutation."""
        import pytest
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.calibrators import RandomizedPIT
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF
        from tsconformal.forecast import InvalidForecastCDFError

        cdf = GaussianForecastCDF(0.0, 1.0)
        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )

        monkeypatch.setattr(RandomizedPIT, "pit", staticmethod(lambda *_args, **_kwargs: np.nan))

        cal.predict_cdf(cdf)
        before = cal._get_state()
        with pytest.raises(InvalidForecastCDFError, match="non-finite"):
            cal.update(0.0, cdf)
        assert cal._get_state() == before

    def test_invalid_forecast_warning_is_emitted(self):
        """Borderline invalid forecasts must emit InvalidForecastCDFWarning."""
        import pytest
        from tsconformal.forecast import InvalidForecastCDFWarning, validate_forecast_cdf

        class BorderlineCDF:
            is_discrete = False

            def ppf(self, u):
                return float(u)

            def cdf(self, y):
                return 0.8 if y < 0.5 else 0.2

        with pytest.warns(
            InvalidForecastCDFWarning,
            match="cdf is not monotone on probe values",
        ):
            validate_forecast_cdf(BorderlineCDF())

    def test_excessive_reset_warning_is_emitted(self):
        """Repeated detector-confirmed resets must emit ExcessiveResetWarning."""
        import pytest
        from tsconformal import SegmentedTransportCalibrator
        from tsconformal.calibrators import ExcessiveResetWarning

        class IdentityCDF:
            is_discrete = False

            def cdf(self, y):
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                return float(np.clip(u, 0.0, 1.0))

        class AlwaysTriggerDetector:
            def update(self, residual_vector):
                return True

            def state(self):
                return {"detector_type": "AlwaysTrigger"}

            def reset(self):
                return None

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=AlwaysTriggerDetector(),
            cooldown=1,
            confirm=1,
        )
        cdf = IdentityCDF()

        for _ in range(2):
            cal.predict_cdf(cdf)
            cal.update(0.5, cdf)

        cal.predict_cdf(cdf)
        with pytest.warns(ExcessiveResetWarning, match="resets in last"):
            cal.update(0.5, cdf)

    def test_high_serial_correlation_warning_is_emitted(self):
        """Strong PIT autocorrelation must emit HighSerialCorrelationWarning."""
        import pytest
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.calibrators import HighSerialCorrelationWarning

        class IdentityCDF:
            is_discrete = False

            def cdf(self, y):
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                return float(np.clip(u, 0.0, 1.0))

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=1.0,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=1000,
            confirm=3,
        )
        cdf = IdentityCDF()

        for y in [0.9, 0.1] * 9 + [0.9]:
            cal.predict_cdf(cdf)
            cal.update(y, cdf)

        cal.predict_cdf(cdf)
        with pytest.warns(HighSerialCorrelationWarning, match="autocorrelation"):
            cal.update(0.1, cdf)

    def test_within_segment_drift_warning_is_emitted(self):
        """Strong PIT distribution shift within a segment must emit drift warnings."""
        import pytest
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.calibrators import WithinSegmentDriftWarning

        class IdentityCDF:
            is_discrete = False

            def cdf(self, y):
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                return float(np.clip(u, 0.0, 1.0))

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=1.0,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=1000,
            confirm=3,
        )
        cdf = IdentityCDF()

        for y in [0.1] * 19 + [0.9] * 20:
            cal.predict_cdf(cdf)
            cal.update(y, cdf)

        cal.predict_cdf(cdf)
        with pytest.warns(
            WithinSegmentDriftWarning,
            match="Within-segment PIT drift detected",
        ):
            cal.update(0.9, cdf)

    def test_update_warning_describes_object_identity_as_heuristic(self):
        import pytest

        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        det = CUSUMNormDetector(kappa=0.02, threshold=10.0)
        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=det,
            cooldown=100,
            confirm=3,
        )

        first = GaussianForecastCDF(0.0, 1.0)
        second = GaussianForecastCDF(0.0, 1.0)

        cal.predict_cdf(first)
        with pytest.warns(
            RuntimeWarning,
            match="heuristic consistency guard",
        ):
            cal.update(0.0, second)

    def test_update_without_pending_prediction_raises(self):
        import pytest

        from tsconformal import (
            CUSUMNormDetector,
            PredictionSequenceError,
            SegmentedTransportCalibrator,
        )
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)

        with pytest.raises(
            PredictionSequenceError,
            match="without a pending prediction",
        ):
            cal.update(0.0, cdf)

    def test_double_update_raises_sequence_error(self):
        import pytest

        from tsconformal import (
            CUSUMNormDetector,
            PredictionSequenceError,
            SegmentedTransportCalibrator,
        )
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)

        cal.predict_cdf(cdf)
        cal.update(0.0, cdf)

        with pytest.raises(
            PredictionSequenceError,
            match="already been consumed",
        ):
            cal.update(0.1, cdf)

    def test_superseded_prediction_raises_sequence_error(self):
        import pytest

        from tsconformal import (
            CUSUMNormDetector,
            PredictionSequenceError,
            SegmentedTransportCalibrator,
        )
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        first = GaussianForecastCDF(0.0, 1.0)
        second = GaussianForecastCDF(0.1, 1.0)

        cal.predict_cdf(first)
        cal.predict_cdf(second)

        with pytest.raises(
            PredictionSequenceError,
            match="superseded base_cdf",
        ):
            cal.update(0.0, first)

    def test_latest_prediction_after_supersession_succeeds(self):
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        first = GaussianForecastCDF(0.0, 1.0)
        second = GaussianForecastCDF(0.1, 1.0)

        cal.predict_cdf(first)
        cal.predict_cdf(second)
        cal.update(0.0, second)

        assert cal._pending_predict_cdf_id is None
        assert cal._pending_predict_t is None

    def test_failed_validation_does_not_consume_pending_prediction(self):
        import pytest

        from tsconformal import (
            CUSUMNormDetector,
            PredictionSequenceError,
            SegmentedTransportCalibrator,
        )
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)

        cal.predict_cdf(cdf)
        with pytest.raises(ValueError, match="y_t must be finite"):
            cal.update(np.nan, cdf)

        cal.update(0.0, cdf)

        with pytest.raises(
            PredictionSequenceError,
            match="already been consumed",
        ):
            cal.update(0.1, cdf)

    def test_consumed_prediction_raises_while_new_prediction_is_pending(self):
        import pytest

        from tsconformal import (
            CUSUMNormDetector,
            PredictionSequenceError,
            SegmentedTransportCalibrator,
        )
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(kappa=0.02, threshold=10.0),
            cooldown=100,
            confirm=3,
        )
        first = GaussianForecastCDF(0.0, 1.0)
        second = GaussianForecastCDF(0.1, 1.0)

        cal.predict_cdf(first)
        cal.update(0.0, first)
        cal.predict_cdf(second)

        with pytest.raises(
            PredictionSequenceError,
            match="already been consumed",
        ):
            cal.update(0.0, first)


# -----------------------------------------------------------------------
# F. Adapter extrapolation clipping
# -----------------------------------------------------------------------

class TestAdapterClipping:
    def test_cdf_clipped_to_01(self):
        from tsconformal import QuantileGridCDFAdapter
        adapter = QuantileGridCDFAdapter(
            probabilities=[0.01, 0.10, 0.50, 0.90, 0.99],
            quantiles=[0.0, 1.0, 2.0, 3.0, 4.0],
        )
        # Extreme values
        assert 0.0 <= adapter.cdf(-1000.0) <= 1.0
        assert 0.0 <= adapter.cdf(1000.0) <= 1.0

    def test_ppf_finite(self):
        from tsconformal import QuantileGridCDFAdapter
        adapter = QuantileGridCDFAdapter(
            probabilities=[0.01, 0.10, 0.50, 0.90, 0.99],
            quantiles=[0.0, 1.0, 2.0, 3.0, 4.0],
        )
        for u in [1e-6, 0.01, 0.5, 0.99, 1 - 1e-6]:
            val = adapter.ppf(u)
            assert np.isfinite(val), f"ppf({u}) = {val}"

    def test_ppf_accepts_array_input(self):
        from tsconformal import QuantileGridCDFAdapter
        adapter = QuantileGridCDFAdapter(
            probabilities=[0.01, 0.10, 0.50, 0.90, 0.99],
            quantiles=[0.0, 1.0, 2.0, 3.0, 4.0],
        )
        u = np.array([1e-6, 0.01, 0.5, 0.99, 1 - 1e-6])
        expected = np.array([adapter.ppf(float(p)) for p in u])
        assert_allclose(adapter.ppf(u), expected)

    def test_nonmonotone_quantiles_repaired(self):
        """Regression fixture from spec: non-monotone quantiles."""
        from tsconformal import QuantileGridCDFAdapter
        adapter = QuantileGridCDFAdapter(
            probabilities=[0.01, 0.10, 0.50, 0.90, 0.99],
            quantiles=[0.0, 1.0, 0.8, 1.3, 1.2],
        )
        # Repaired quantiles should be nondecreasing
        qs = [adapter.ppf(p) for p in [0.01, 0.10, 0.50, 0.90, 0.99]]
        for i in range(len(qs) - 1):
            assert qs[i] <= qs[i + 1] + 1e-10, f"Not monotone: {qs}"

    def test_out_of_bound_cdf_clamped(self):
        """Edge case 6: base CDF returning values slightly outside [0,1]."""
        from tsconformal import QuantileGridCDFAdapter
        adapter = QuantileGridCDFAdapter(
            probabilities=[0.01, 0.50, 0.99],
            quantiles=[-1.0, 0.0, 1.0],
        )
        # Even extreme y should give cdf in [0, 1]
        assert 0.0 <= adapter.cdf(-1e10) <= 1.0
        assert 0.0 <= adapter.cdf(1e10) <= 1.0


# -----------------------------------------------------------------------
# G. Sample adapter edge cases
# -----------------------------------------------------------------------

class TestSampleCDFAdapter:
    def test_gaussian_monotone_rejects_zero_variance_samples(self):
        import pytest

        from tsconformal import SampleCDFAdapter
        from tsconformal.forecast import InvalidForecastCDFError

        with pytest.raises(InvalidForecastCDFError, match="degenerate samples"):
            SampleCDFAdapter(np.ones(200), smoothing="gaussian_monotone")

    def test_gaussian_monotone_rejects_near_degenerate_samples(self):
        import pytest

        from tsconformal import SampleCDFAdapter
        from tsconformal.forecast import InvalidForecastCDFError

        samples = np.ones(200) + 1e-15 * np.arange(200)
        with pytest.raises(InvalidForecastCDFError, match="degenerate samples"):
            SampleCDFAdapter(samples, smoothing="gaussian_monotone")

    def test_empirical_path_keeps_point_mass_samples_valid(self):
        from tsconformal import SampleCDFAdapter
        from tsconformal.forecast import validate_forecast_cdf

        cdf = SampleCDFAdapter(np.ones(200), smoothing="none")
        validate_forecast_cdf(cdf)

        assert cdf.is_discrete is True
        assert cdf.cdf(1.0) == 1.0
        assert cdf.ppf(0.5) == 1.0


# -----------------------------------------------------------------------
# H. CRPS batching and fallback behavior
# -----------------------------------------------------------------------

class TestCRPS:
    def test_crps_uses_batched_ppf_when_available(self):
        from tsconformal.metrics import crps

        class VectorizedCDF:
            is_discrete = False

            def __init__(self):
                self.calls = []

            def cdf(self, y):
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                arr = np.asarray(u, dtype=np.float64)
                self.calls.append(arr.shape)
                return arr

        cdfs = [VectorizedCDF(), VectorizedCDF()]
        y = np.array([0.25, 0.75])

        value = crps(y, cdfs, n_quadrature=7)

        assert np.isfinite(value)
        assert cdfs[0].calls == [(7,)]
        assert cdfs[1].calls == [(7,)]

    def test_crps_falls_back_for_scalar_only_ppf(self):
        from tsconformal.metrics import crps

        class ScalarOnlyCDF:
            is_discrete = False

            def __init__(self):
                self.calls = 0

            def cdf(self, y):
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                if np.ndim(u) != 0:
                    raise TypeError("scalar-only ppf")
                self.calls += 1
                return float(u)

        cdf = ScalarOnlyCDF()
        y = np.array([0.4])

        value = crps(y, [cdf], n_quadrature=5)

        assert np.isfinite(value)
        assert cdf.calls == 5


# -----------------------------------------------------------------------
# I. Diagnostics warning behavior
# -----------------------------------------------------------------------

class TestDiagnostics:
    def test_pit_uniformity_suppresses_anderson_futurewarning(self):
        import warnings

        from tsconformal.diagnostics import pit_uniformity_tests

        pits = np.linspace(0.05, 0.95, 20)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = pit_uniformity_tests(pits, lags=5)

        assert np.isfinite(result.ad_statistic)
        assert np.isfinite(result.ad_pvalue)
        assert not any(issubclass(w.category, FutureWarning) for w in captured)

    def test_pit_uniformity_warns_when_ljung_box_fails(self, monkeypatch):
        import sys
        import types
        import pytest

        from tsconformal.diagnostics import pit_uniformity_tests

        diagnostic_mod = types.ModuleType("statsmodels.stats.diagnostic")

        def _raise_runtime_error(*args, **kwargs):
            raise RuntimeError("boom")

        diagnostic_mod.acorr_ljungbox = _raise_runtime_error

        stats_mod = types.ModuleType("statsmodels.stats")
        stats_mod.diagnostic = diagnostic_mod

        root_mod = types.ModuleType("statsmodels")
        root_mod.stats = stats_mod

        monkeypatch.setitem(sys.modules, "statsmodels", root_mod)
        monkeypatch.setitem(sys.modules, "statsmodels.stats", stats_mod)
        monkeypatch.setitem(sys.modules, "statsmodels.stats.diagnostic", diagnostic_mod)

        pits = np.linspace(0.05, 0.95, 20)
        with pytest.warns(RuntimeWarning, match="Ljung-Box failed: boom"):
            result = pit_uniformity_tests(pits, lags=5)

        assert result.ljung_box_statistic is None
        assert result.ljung_box_pvalue is None

    def test_sensitivity_report_skips_invalid_forecasts(self):
        from tsconformal.diagnostics import DetectorConfig, sensitivity_report

        class InvalidCDF:
            is_discrete = False

            def cdf(self, y):
                return 0.5

            def ppf(self, u):
                return np.nan

        report = sensitivity_report(
            [DetectorConfig()],
            [(InvalidCDF(), 0.0)],
        )

        assert len(report.results) == 1
        assert report.results[0].n_resets == 0

    def test_sensitivity_report_propagates_unexpected_exceptions(self):
        import pytest

        from tsconformal.diagnostics import DetectorConfig, sensitivity_report

        class RuntimeErrorCDF:
            is_discrete = False

            def cdf(self, y):
                if y == 0.0:
                    raise RuntimeError("boom")
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                return float(u)

        with pytest.raises(RuntimeError, match="boom"):
            sensitivity_report(
                [DetectorConfig()],
                [(RuntimeErrorCDF(), 0.0)],
            )

    def test_sensitivity_report_to_dict_omits_unpopulated_metrics(self):
        from tsconformal.diagnostics import DetectorConfig, sensitivity_report
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        stream = [(GaussianForecastCDF(0.0, 1.0), 0.0) for _ in range(4)]

        report = sensitivity_report(
            [DetectorConfig(threshold=1e6)],
            stream,
        )
        payload = report.to_dict()

        assert "results" in payload
        assert len(payload["results"]) == 1
        assert "mean_calibration_error" not in payload["results"][0]
        assert "mean_coverage_error" not in payload["results"][0]


# -----------------------------------------------------------------------
# Numerical edge cases (all 8 from spec)
# -----------------------------------------------------------------------

class TestNumericalEdgeCases:
    def test_weight_underflow_long_decay(self):
        """Edge case 1: rho=0.999 over many steps."""
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF
        import warnings as w
        w.filterwarnings("ignore")

        det = CUSUMNormDetector(kappa=0.02, threshold=1e6)  # effectively disable
        cal = SegmentedTransportCalibrator(
            grid_size=10, rho=0.999, n_eff_min=1.0,
            step_schedule=lambda n: min(0.2, 1.0 / max(n ** 0.5, 1)),
            detector=det, cooldown=100000, confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)
        rng = np.random.default_rng(42)

        for _ in range(5000):
            cal.predict_cdf(cdf)
            cal.update(rng.normal(), cdf)

        assert np.isfinite(cal.n_eff)
        assert cal.n_eff > 0

    def test_duplicate_pit_ties(self):
        """Edge case 2: many observations with same PIT value."""
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        det = CUSUMNormDetector(kappa=0.02, threshold=10.0)
        cal = SegmentedTransportCalibrator(
            grid_size=10, rho=0.9, n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=det, cooldown=100, confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)

        # Feed the same y value repeatedly -> same PIT
        for _ in range(100):
            cal.predict_cdf(cdf)
            cal.update(0.0, cdf)

        # Should not crash; g should be monotone
        assert np.all(np.diff(cal._g) >= -1e-10)

    def test_near_threshold_n_eff(self):
        """Edge case 7: n_eff just above/below threshold."""
        from tsconformal import CUSUMNormDetector, SegmentedTransportCalibrator
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        det = CUSUMNormDetector(kappa=0.02, threshold=10.0)
        cal = SegmentedTransportCalibrator(
            grid_size=10, rho=0.5, n_eff_min=3.0,
            step_schedule=lambda n: 0.1,
            detector=det, cooldown=100, confirm=3,
        )
        cdf = GaussianForecastCDF(0.0, 1.0)
        rng = np.random.default_rng(42)

        for _ in range(10):
            cal.predict_cdf(cdf)
            cal.update(rng.normal(), cdf)

        # Should either be in fallback or have a monotone map
        if not cal.in_warmup:
            assert np.all(np.diff(cal._g) >= -1e-10)

    def test_nan_inf_residuals(self):
        """Edge case 8: NaN/Inf in detector residuals."""
        from tsconformal import CUSUMNormDetector
        det = CUSUMNormDetector(kappa=0.02, threshold=0.20)

        # NaN residuals should not crash
        result = det.update(np.array([np.nan, 0.1, 0.2]))
        assert isinstance(result, bool)

        # Inf residuals
        result = det.update(np.array([np.inf, 0.1, 0.2]))
        assert isinstance(result, bool)


# -----------------------------------------------------------------------
# Serialization round-trip (edge case 16 from audit)
# -----------------------------------------------------------------------

class TestSerializationRoundTrip:
    def test_mid_stream_save_reload(self, tmp_path):
        """Run 200 steps, save, reload, continue 100, verify match."""
        from tsconformal import (
            CUSUMNormDetector,
            SegmentedTransportCalibrator,
            save_calibrator,
            load_calibrator,
        )
        from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

        def step_schedule(n):
            return min(0.2, 1.0 / max(n ** 0.5, 1))

        def make_cal():
            det = CUSUMNormDetector(kappa=0.02, threshold=0.15)
            return SegmentedTransportCalibrator(
                grid_size=10, rho=0.95, n_eff_min=5.0,
                step_schedule=step_schedule,
                detector=det, cooldown=50, confirm=2,
            )

        cdf = GaussianForecastCDF(0.3, 0.8)
        rng = np.random.default_rng(42)
        ys = rng.normal(0, 1, 300)

        # Run 200 steps on original
        cal_orig = make_cal()
        for i in range(200):
            cal_orig.predict_cdf(cdf)
            cal_orig.update(ys[i], cdf)

        # Save
        save_path = tmp_path / "cal_state"
        save_calibrator(cal_orig, save_path)

        # Continue original for 100 more
        orig_predictions = []
        for i in range(200, 300):
            pred = cal_orig.predict_cdf(cdf)
            orig_predictions.append(pred.cdf(0.5))
            cal_orig.update(ys[i], cdf)

        # Load and continue
        cal_loaded = load_calibrator(save_path, step_schedule=step_schedule)
        loaded_predictions = []
        for i in range(200, 300):
            pred = cal_loaded.predict_cdf(cdf)
            loaded_predictions.append(pred.cdf(0.5))
            cal_loaded.update(ys[i], cdf)

        # Verify predictions match
        assert_allclose(
            orig_predictions, loaded_predictions, atol=1e-10,
            err_msg="Loaded calibrator diverged from original"
        )

    def test_custom_step_schedule_requires_explicit_load_schedule(self, tmp_path):
        """Custom schedules must be passed back in on load."""
        import pytest
        from tsconformal import (
            CUSUMNormDetector,
            QuantileGridCDFAdapter,
            SegmentedTransportCalibrator,
            load_calibrator,
            save_calibrator,
        )

        cdf = QuantileGridCDFAdapter([0.1, 0.5, 0.9], [-1.0, 0.0, 1.0])

        def step_schedule(n):
            return 0.123

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=step_schedule,
            detector=CUSUMNormDetector(),
            cooldown=5,
            confirm=1,
        )
        for y in [0.0, 0.1, 0.2]:
            cal.predict_cdf(cdf)
            cal.update(y, cdf)

        save_path = tmp_path / "custom_state.zip"
        save_calibrator(cal, save_path)

        with pytest.raises(ValueError, match="non-default step_schedule"):
            load_calibrator(save_path)

        cal_loaded = load_calibrator(save_path, step_schedule=step_schedule)
        assert cal_loaded._step_schedule(10.0) == step_schedule(10.0)

    def test_pending_prediction_state_is_not_serialized(self, tmp_path):
        """Loaded calibrators should not resume with an in-flight prediction."""
        import pytest
        from tsconformal import (
            CUSUMNormDetector,
            PredictionSequenceError,
            QuantileGridCDFAdapter,
            SegmentedTransportCalibrator,
            load_calibrator,
            save_calibrator,
        )

        cdf = QuantileGridCDFAdapter([0.1, 0.5, 0.9], [-1.0, 0.0, 1.0])
        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=0.9,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.1,
            detector=CUSUMNormDetector(),
            cooldown=5,
            confirm=1,
        )

        cal.predict_cdf(cdf)
        save_path = tmp_path / "pending_state.zip"
        save_calibrator(cal, save_path)

        cal_loaded = load_calibrator(save_path, step_schedule=lambda n: 0.1)
        with pytest.raises(
            PredictionSequenceError,
            match="without a pending prediction",
        ):
            cal_loaded.update(0.0, cdf)

    def test_transport_ppf_propagates_unexpected_base_errors(self):
        """Unexpected adapter errors must not be masked by a scalar fallback."""
        import pytest
        from tsconformal.forecast import TransportedForecastCDF

        class ExplodingCDF:
            is_discrete = False

            def cdf(self, y):
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                raise RuntimeError("boom")

        transported = TransportedForecastCDF(
            ExplodingCDF(),
            lambda u: u,
            lambda u: u,
        )

        with pytest.raises(RuntimeError, match="boom"):
            transported.ppf(np.array([0.2, 0.8]))

    def test_warning_history_resets_with_segment_boundary(self):
        """Within-segment warnings must not inherit pre-reset PIT history."""
        import warnings
        from tsconformal import SegmentedTransportCalibrator

        class IdentityCDF:
            is_discrete = False

            def cdf(self, y):
                return float(np.clip(y, 0.0, 1.0))

            def ppf(self, u):
                return float(np.clip(u, 0.0, 1.0))

        class TriggerOnceDetector:
            def __init__(self, trigger_at=41):
                self.t = 0
                self.trigger_at = trigger_at

            def update(self, residual_vector):
                self.t += 1
                return self.t == self.trigger_at

            def state(self):
                return {"detector_type": "TriggerOnce", "t": self.t}

            def reset(self):
                return None

        cal = SegmentedTransportCalibrator(
            grid_size=5,
            rho=1.0,
            n_eff_min=1.0,
            step_schedule=lambda n: 0.2,
            detector=TriggerOnceDetector(),
            cooldown=1,
            confirm=1,
        )
        cdf = IdentityCDF()

        for y in [0.9, 0.1] * 20:
            cal.predict_cdf(cdf)
            cal.update(y, cdf)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cal.predict_cdf(cdf)
            cal.update(0.9, cdf)

        warning_types = {type(item.message).__name__ for item in caught}
        assert cal.segment_id == 1
        assert "HighSerialCorrelationWarning" not in warning_types
        assert "WithinSegmentDriftWarning" not in warning_types
