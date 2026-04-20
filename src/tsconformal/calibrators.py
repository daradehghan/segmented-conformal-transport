"""Segmented Conformal Transport calibrator and supporting internals.

This module provides ``SegmentedTransportCalibrator``,
``RandomizedPIT``, and the save/load helpers. It contains the online
state recursion, isotonic map updates, detector-triggered resets, and
identity fallback used by SCT.
"""

from __future__ import annotations

import json
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, cast

import numpy as np
from scipy.stats import ks_2samp

from tsconformal.detectors import SegmentDetector
from tsconformal.forecast import (
    ForecastCDF,
    InvalidForecastCDFError,
    TransportedForecastCDF,
    validate_forecast_cdf,
)
from tsconformal.utils import (
    build_transport_map,
    clip_cdf,
    ensure_rng,
    pav_isotonic,
)


# -----------------------------------------------------------------------
# Warning classes
# -----------------------------------------------------------------------

class LowEffectiveSampleWarning(UserWarning):
    """n_eff < n_eff_min; identity fallback activated."""


class ExcessiveResetWarning(UserWarning):
    """Too many resets in a short window."""


class HighSerialCorrelationWarning(UserWarning):
    """PIT residuals show high serial correlation; mixing may be violated."""


class WithinSegmentDriftWarning(UserWarning):
    """Rolling PIT tests suggest within-segment drift."""


class WarmStartDominanceWarning(UserWarning):
    """Post-reset warm-start weight is large."""


class DiscreteForecastWithoutRandomizedPITError(Exception):
    """Discrete forecast used without randomized PIT support."""


class PredictionSequenceError(RuntimeError):
    """Invalid ``predict_cdf()``/``update()`` call sequence."""


# -----------------------------------------------------------------------
# RandomizedPIT
# -----------------------------------------------------------------------

class RandomizedPIT:
    """Compute continuous or randomized probability integral transforms.

    If the forecast is continuous at y, returns the standard PIT
    U_t = F_hat(y). If the forecast is discrete (has an atom at y),
    draws an auxiliary uniform V_t and returns the randomized PIT
    on the jump interval.

    This class does not mutate calibrator state.
    """

    @staticmethod
    def pit(cdf: ForecastCDF, y: float, rng=None) -> float:
        """Compute the (possibly randomized) PIT value.

        Parameters
        ----------
        cdf : ForecastCDF
            Base predictive CDF.
        y : float
            Observed outcome.
        rng : numpy Generator or seed, optional
            Random number generator for randomization.

        Returns
        -------
        float in [0, 1]
        """
        if not cdf.is_discrete:
            return float(clip_cdf(cdf.cdf(y)))

        # Discrete case: randomized PIT
        rng = ensure_rng(rng)
        F_y = float(cdf.cdf(y))
        cdf_left = getattr(cdf, "cdf_left", None)
        if callable(cdf_left):
            F_y_minus = float(cdf_left(y))
        else:
            y_minus = np.nextafter(float(y), -np.inf)
            F_y_minus = float(cdf.cdf(y_minus))

        jump = F_y - F_y_minus
        if jump < 1e-15:
            return float(clip_cdf(F_y))

        V = rng.uniform()
        U = F_y_minus + V * jump
        return float(clip_cdf(U))


# -----------------------------------------------------------------------
# SegmentedTransportCalibrator
# -----------------------------------------------------------------------

def _default_step(n: float) -> float:
    return min(0.20, 1.0 / max(np.sqrt(n), 1e-8))


def _identity_transport(u):
    return u


_TAIL_LEN = 256
_DEFAULT_STEP_SCHEDULE_ID = "__default__"
_CUSTOM_STEP_SCHEDULE_ID = "__custom__"

# Serialized state has its own compatibility contract. The package version may
# advance without changing the persisted layout.
STATE_SCHEMA_VERSION = "0.1.0"
SUPPORTED_STATE_SCHEMA_VERSIONS = {STATE_SCHEMA_VERSION}


def _resolve_step_schedule_id(step_schedule: Callable[[float], float]) -> str:
    """Return a stable identifier for the step schedule."""
    if step_schedule is _default_step:
        return _DEFAULT_STEP_SCHEDULE_ID

    schedule_id = getattr(step_schedule, "_schedule_name", None)
    if schedule_id is not None:
        return str(schedule_id)

    return _CUSTOM_STEP_SCHEDULE_ID


def _validate_state_schema_version(state_schema_version: Any) -> str:
    """Accept only explicit schema versions that we know how to load."""
    if state_schema_version not in SUPPORTED_STATE_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported state schema version {state_schema_version!r}; "
            f"supported versions are {sorted(SUPPORTED_STATE_SCHEMA_VERSIONS)!r}."
        )
    return str(state_schema_version)


class SegmentedTransportCalibrator:
    """Segmented Conformal Transport (SCT) calibrator.

    This is the main public class implementing the online PIT-transport
    recalibration algorithm from the verified theory.

    Parameters
    ----------
    grid_size : int
        Number of interior grid points J.
    rho : float
        Recency discount parameter in (0, 1].
    n_eff_min : float
        Minimum effective sample size before emitting non-fallback output.
    step_schedule : callable
        Maps n_eff -> eta_t in (0, 1).
    detector : SegmentDetector
        Change-point detector instance.
    cooldown : int
        Minimum segment length before a reset is allowed.
    confirm : int
        Number of consecutive detector threshold exceedances before reset.
    projection : str
        Must be 'isotonic_l2'.
    interpolation : str
        Must be 'piecewise_linear'.
    fallback : str
        Must be 'identity'.
    warm_start : str
        Must be 'identity'.
    """

    def __init__(
        self,
        grid_size: int,
        rho: float,
        n_eff_min: float,
        step_schedule: Callable[[float], float],
        detector: SegmentDetector,
        cooldown: int,
        confirm: int,
        projection: Literal["isotonic_l2"] = "isotonic_l2",
        interpolation: Literal["piecewise_linear"] = "piecewise_linear",
        fallback: Literal["identity"] = "identity",
        warm_start: Literal["identity"] = "identity",
    ):
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if not (0.0 < rho <= 1.0):
            raise ValueError("rho must be in (0, 1]")
        if n_eff_min < 1.0:
            raise ValueError("n_eff_min must be >= 1")
        if cooldown < 1:
            raise ValueError("cooldown must be >= 1")
        if confirm < 1:
            raise ValueError("confirm must be >= 1")
        if projection != "isotonic_l2":
            raise ValueError(
                f"unsupported projection: {projection!r}; only 'isotonic_l2' "
                "is supported."
            )
        if interpolation != "piecewise_linear":
            raise ValueError(
                f"unsupported interpolation: {interpolation!r}; only "
                "'piecewise_linear' is supported."
            )
        if fallback != "identity":
            raise ValueError(
                f"unsupported fallback: {fallback!r}; only 'identity' is "
                "supported."
            )
        if warm_start != "identity":
            raise ValueError(
                f"unsupported warm_start: {warm_start!r}; only 'identity' is "
                "supported."
            )

        self._grid_size = grid_size
        self._rho = rho
        self._n_eff_min = n_eff_min
        self._step_schedule = step_schedule
        self._detector = detector
        self._cooldown = cooldown
        self._confirm_len = confirm
        self._projection = projection
        self._interpolation = interpolation
        self._fallback_mode = fallback
        self._warm_start_mode = warm_start

        # Grid: u_j = j / (J + 1), j = 1, ..., J
        J = grid_size
        self._grid = np.array([j / (J + 1) for j in range(1, J + 1)])

        # Initialize state
        self._reset_state(initial=True)
        self._reset_prediction_sequence_state()

    def _reset_state(self, initial: bool = False) -> None:
        """Reset all segment-local state."""
        J = self._grid_size
        self._W: float = 0.0
        self._W2: float = 0.0
        self._N: np.ndarray = np.zeros(J, dtype=np.float64)
        self._m: np.ndarray = self._grid.copy()  # identity warm start
        self._g: np.ndarray = self._grid.copy()  # identity warm start
        self._confirm_count: int = 0
        self._in_fallback: bool = True
        self._warm_start_mass: float = 1.0  # starts at 1.0 (pure identity)
        self._t: int = 0 if initial else self._t
        self._segment_start: int = self._t
        self._segment_id: int = 0 if initial else self._segment_id + 1
        self._num_resets: int = 0 if initial else self._num_resets + 1
        self._last_reset_t: int = self._t

        # Diagnostics
        self._pit_history: deque = deque(maxlen=_TAIL_LEN)
        self._residual_history: deque = deque(maxlen=_TAIL_LEN)
        self._warnings_list: List[str] = []
        if initial:
            self._reset_times: deque = deque(maxlen=_TAIL_LEN)
        # Don't clear reset_times on segment reset — it's global

    def _reset_prediction_sequence_state(self) -> None:
        """Initialize transient predict/update sequencing state."""
        self._pending_predict_cdf_id: int | None = None
        self._pending_predict_t: int | None = None
        self._last_consumed_predict_cdf_id: int | None = None
        self._superseded_predict_cdf_ids: set[int] = set()

    def _consume_pending_prediction(self) -> None:
        """Mark the current pending prediction as consumed."""
        self._last_consumed_predict_cdf_id = self._pending_predict_cdf_id
        self._pending_predict_cdf_id = None
        self._pending_predict_t = None
        self._superseded_predict_cdf_ids.clear()

    # -------------------------------------------------------------------
    # Public properties (Appendix D diagnostics)
    # -------------------------------------------------------------------

    @property
    def segment_id(self) -> int:
        return self._segment_id

    @property
    def num_resets(self) -> int:
        return self._num_resets

    @property
    def last_reset_t(self) -> int:
        return self._last_reset_t

    @property
    def n_eff(self) -> float:
        if self._W2 < 1e-15:
            return 0.0
        return self._W ** 2 / self._W2

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def grid(self) -> np.ndarray:
        return self._grid.copy()

    @property
    def in_warmup(self) -> bool:
        return self._in_fallback

    @property
    def warm_start_weight(self) -> float:
        """Exact surviving identity contribution from the smoothing recursion.

        Under the recursion m_t = (1-eta_t) m_{t-1} + eta_t G_hat_t,
        after reset the warm-start weight is the product of (1-eta_q)
        over all active (non-fallback) steps since the last reset.
        """
        return self._warm_start_mass

    @property
    def pit_history_tail(self) -> np.ndarray:
        return np.array(list(self._pit_history), dtype=np.float64)

    @property
    def calibration_residuals_tail(self) -> np.ndarray:
        return np.array(list(self._residual_history), dtype=np.float64)

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings_list)

    # -------------------------------------------------------------------
    # Main public methods
    # -------------------------------------------------------------------

    def predict_cdf(
        self,
        base_cdf: ForecastCDF,
        x_t=None,
    ) -> TransportedForecastCDF:
        """Produce a calibrated predictive CDF for the current time step.

        This method does NOT consume y_t — it uses only the current
        transport map built from past data. The base forecast is validated
        before consumption.

        Parameters
        ----------
        base_cdf : ForecastCDF
            Base forecast for the current time step.
        x_t : ignored
            Placeholder for covariates (unused in the current implementation).

        Returns
        -------
        TransportedForecastCDF
            Calibrated forecast supporting cdf() and ppf().

        Notes
        -----
        The method records a pending prediction for the current time step.
        A later ``predict_cdf()`` call supersedes any earlier pending
        prediction before ``update()`` consumes it. The method does not
        consume ``y_t`` or change the transport map itself.

        Raises
        ------
        InvalidForecastCDFError
            If the base forecast fails validation.
        """
        # Validate before consuming
        validate_forecast_cdf(base_cdf)

        if self._pending_predict_cdf_id is not None:
            self._superseded_predict_cdf_ids.add(self._pending_predict_cdf_id)

        pending_cdf_id = id(base_cdf)
        self._pending_predict_cdf_id = pending_cdf_id
        self._pending_predict_t = self._t
        self._superseded_predict_cdf_ids.discard(pending_cdf_id)

        if self._in_fallback:
            T = _identity_transport
            T_inv = _identity_transport
        else:
            T, T_inv = build_transport_map(self._grid, self._g)

        return TransportedForecastCDF(base_cdf, T, T_inv)

    def update(
        self,
        y_t: float,
        base_cdf: ForecastCDF,
        rng=None,
    ) -> None:
        """Update the calibrator state after observing the outcome.

        This should be called with the forecast used to issue the
        preceding ``predict_cdf`` call.

        Parameters
        ----------
        y_t : float
            Observed scalar outcome.
        base_cdf : ForecastCDF
            The base forecast that generated the prediction.
        rng : numpy Generator or seed, optional
            RNG for randomized PIT (required if base_cdf.is_discrete).

        Notes
        -----
        The implementation raises ``PredictionSequenceError`` on clear
        sequencing violations such as ``update()`` without a pending
        prediction, repeated ``update()`` after a prediction has already
        been consumed, or use of a superseded pending forecast. It still
        warns on ambiguous forecast-object mismatches, because object
        identity is only a heuristic consistency check.
        """
        base_cdf_id = id(base_cdf)
        pending_cdf_id = self._pending_predict_cdf_id

        if pending_cdf_id is None:
            if base_cdf_id == self._last_consumed_predict_cdf_id:
                raise PredictionSequenceError(
                    "update() called after the pending prediction had already been "
                    "consumed; call predict_cdf() before updating again."
                )
            raise PredictionSequenceError(
                "update() called without a pending prediction; call "
                "predict_cdf() before update()."
            )

        if base_cdf_id in self._superseded_predict_cdf_ids:
            raise PredictionSequenceError(
                "update() called with a superseded base_cdf; use the most "
                f"recent forecast issued by predict_cdf() for t={self._pending_predict_t}."
            )

        if (
            base_cdf_id == self._last_consumed_predict_cdf_id
            and base_cdf_id != pending_cdf_id
        ):
            raise PredictionSequenceError(
                "update() called with a prediction that has already been "
                "consumed; call predict_cdf() before updating again."
            )

        if base_cdf_id != pending_cdf_id:
            warnings.warn(
                "update() called with a different base_cdf than predict_cdf() "
                f"for the current pending prediction at t={self._pending_predict_t}. "
                "Reuse the forecast used to issue the prediction; this "
                "object-identity check is only a heuristic consistency guard.",
                RuntimeWarning,
                stacklevel=2,
            )

        if not np.isfinite(y_t):
            raise ValueError("y_t must be finite")

        # Check discrete forecast without RNG
        if base_cdf.is_discrete and rng is None:
            raise DiscreteForecastWithoutRandomizedPITError(
                "Discrete forecast requires rng for randomized PIT"
            )

        # Compute PIT
        U_t = RandomizedPIT.pit(base_cdf, y_t, rng)
        if not np.isfinite(U_t):
            raise InvalidForecastCDFError(f"RandomizedPIT.pit returned non-finite: {U_t}")
        if U_t < 0.0 or U_t > 1.0:
            raise InvalidForecastCDFError(f"RandomizedPIT.pit returned out-of-range value: {U_t}")

        self._t += 1
        self._warnings_list.clear()
        self._pit_history.append(U_t)

        # Compute detector residuals: r_{t,j} = 1{U_t <= u_j} - g_{j}
        indicators = (U_t <= self._grid).astype(np.float64)
        residuals = indicators - self._g
        self._residual_history.append(residuals.copy())

        # Update detector
        exceeded = self._detector.update(residuals)

        # Confirmation logic
        if exceeded:
            self._confirm_count += 1
        else:
            self._confirm_count = 0

        # Check reset conditions
        segment_len = self._t - self._segment_start
        if self._confirm_count >= self._confirm_len and segment_len >= self._cooldown:
            self._do_reset()
            # After reset, keep processing the current observation so it
            # seeds the new archive before the aggregate update below.

        # Update recency-weighted aggregates
        self._W = self._rho * self._W + 1.0
        self._W2 = self._rho ** 2 * self._W2 + 1.0
        self._N = self._rho * self._N + indicators

        # Raw weighted ECDF estimate
        G_hat = self._N / max(self._W, 1e-15)

        # Compute n_eff
        current_n_eff = self.n_eff

        # Check fallback
        if current_n_eff < self._n_eff_min:
            self._m = self._grid.copy()
            self._g = self._grid.copy()
            self._in_fallback = True
            self._warnings_list.append("LowEffectiveSample")
            warnings.warn(
                f"n_eff={current_n_eff:.1f} < n_eff_min={self._n_eff_min}; "
                "identity fallback activated",
                LowEffectiveSampleWarning,
                stacklevel=2,
            )
            self._consume_pending_prediction()
            return

        # Feedback smoothing: convex combination
        eta_t = self._step_schedule(current_n_eff)
        eta_t = max(1e-10, min(eta_t, 0.999))
        self._m = (1.0 - eta_t) * self._m + eta_t * G_hat
        self._warm_start_mass *= (1.0 - eta_t)  # decay identity contribution

        # Isotonic L2 projection
        self._g = pav_isotonic(self._m)

        self._in_fallback = False

        # Misuse warnings
        self._check_warnings()
        self._consume_pending_prediction()

    def _do_reset(self) -> None:
        """Execute a confirmed reset."""
        self._W = 0.0
        self._W2 = 0.0
        self._N = np.zeros(self._grid_size, dtype=np.float64)
        self._m = self._grid.copy()
        self._g = self._grid.copy()
        self._confirm_count = 0
        self._in_fallback = True
        self._warm_start_mass = 1.0  # reset to pure identity
        self._segment_start = self._t
        self._segment_id += 1
        self._num_resets += 1
        self._last_reset_t = self._t
        self._reset_times.append(self._t)  # track for rolling window warning
        self._pit_history.clear()
        self._residual_history.clear()

        # Reset detector internal state (required by SegmentDetector protocol)
        self._detector.reset()

    def _check_warnings(self) -> None:
        """Check for misuse patterns and emit spec §3.6 warnings."""
        # 1. ExcessiveResetWarning: rolling window, not lifetime average
        rolling_window = max(self._cooldown * 5, 500)
        recent_resets = sum(
            1 for rt in self._reset_times
            if self._t - rt <= rolling_window
        )
        if recent_resets >= 3:  # 3+ resets in rolling window
            self._warnings_list.append("ExcessiveReset")
            warnings.warn(
                f"{recent_resets} resets in last {rolling_window} steps",
                ExcessiveResetWarning,
                stacklevel=3,
            )

        # 2. HighSerialCorrelationWarning: within-segment PIT residuals
        segment_pits = [
            p for i, p in enumerate(self._pit_history)
            if i >= max(0, len(self._pit_history) - 50)
        ]
        if len(segment_pits) >= 20:
            centered = np.array(segment_pits) - 0.5
            if np.std(centered) > 1e-10:
                lag1_corr = np.corrcoef(centered[:-1], centered[1:])[0, 1]
                if np.isfinite(lag1_corr) and abs(lag1_corr) > 0.5:
                    self._warnings_list.append("HighSerialCorrelation")
                    warnings.warn(
                        f"High lag-1 autocorrelation in within-segment PITs: "
                        f"{lag1_corr:.3f}",
                        HighSerialCorrelationWarning,
                        stacklevel=3,
                    )

        # 3. WithinSegmentDriftWarning: compare first/second half PITs
        if len(segment_pits) >= 40:
            half = len(segment_pits) // 2
            first_half = np.array(segment_pits[:half])
            second_half = np.array(segment_pits[half:])
            # Two-sample KS between first and second half
            _, drift_p = ks_2samp(first_half, second_half)
            if drift_p < 0.01:  # strong evidence of within-segment drift
                self._warnings_list.append("WithinSegmentDrift")
                warnings.warn(
                    f"Within-segment PIT drift detected (KS p={drift_p:.4f})",
                    WithinSegmentDriftWarning,
                    stacklevel=3,
                )

        # 4. WarmStartDominanceWarning: exact warm-start weight
        if self._warm_start_mass > 0.5 and not self._in_fallback:
            self._warnings_list.append("WarmStartDominance")
            warnings.warn(
                f"Post-reset warm-start weight {self._warm_start_mass:.3f} > 0.5; "
                "calibration map still dominated by identity initialization",
                WarmStartDominanceWarning,
                stacklevel=3,
            )

    # -------------------------------------------------------------------
    # State access for serialization
    # -------------------------------------------------------------------

    def _get_state(self) -> dict:
        """Return full internal state as a serializable dict."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "grid_size": self._grid_size,
            "rho": self._rho,
            "n_eff_min": self._n_eff_min,
            "cooldown": self._cooldown,
            "confirm": self._confirm_len,
            "projection": self._projection,
            "interpolation": self._interpolation,
            "fallback": self._fallback_mode,
            "warm_start": self._warm_start_mode,
            "step_schedule_id": _resolve_step_schedule_id(self._step_schedule),
            "t": self._t,
            "segment_id": self._segment_id,
            "num_resets": self._num_resets,
            "last_reset_t": self._last_reset_t,
            "segment_start": self._segment_start,
            "W": self._W,
            "W2": self._W2,
            "N": self._N.tolist(),
            "m": self._m.tolist(),
            "g": self._g.tolist(),
            "grid": self._grid.tolist(),
            "confirm_count": self._confirm_count,
            "in_fallback": self._in_fallback,
            "warm_start_mass": self._warm_start_mass,
            "detector_type": type(self._detector).__name__,
            "detector_module": type(self._detector).__module__,
            "detector_state": dict(self._detector.state()),
            "pit_history": list(self._pit_history),
            "residual_history": [r.tolist() for r in self._residual_history],
            "reset_times": list(self._reset_times),
            "warnings": list(self._warnings_list),
        }

    def _set_state(self, state: dict) -> None:
        """Restore internal state from a dict."""
        _validate_state_schema_version(state.get("schema_version"))
        self._reset_prediction_sequence_state()
        self._t = state["t"]
        self._segment_id = state["segment_id"]
        self._num_resets = state["num_resets"]
        self._last_reset_t = state["last_reset_t"]
        self._segment_start = state["segment_start"]
        self._W = state["W"]
        self._W2 = state["W2"]
        self._N = np.array(state["N"], dtype=np.float64)
        self._m = np.array(state["m"], dtype=np.float64)
        self._g = np.array(state["g"], dtype=np.float64)
        self._confirm_count = state["confirm_count"]
        self._in_fallback = state["in_fallback"]
        self._warm_start_mass = state.get("warm_start_mass", 1.0)
        self._pit_history = deque(state.get("pit_history", []), maxlen=_TAIL_LEN)
        self._residual_history = deque(
            [np.array(r) for r in state.get("residual_history", [])],
            maxlen=_TAIL_LEN,
        )
        self._warnings_list = list(state.get("warnings", []))
        self._reset_times = deque(state.get("reset_times", []), maxlen=_TAIL_LEN)

    def __repr__(self) -> str:
        return (
            f"SegmentedTransportCalibrator("
            f"J={self._grid_size}, rho={self._rho}, "
            f"n_eff={self.n_eff:.1f}, segment={self._segment_id}, "
            f"resets={self._num_resets}, t={self._t})"
        )


# -----------------------------------------------------------------------
# Serialization helpers
# -----------------------------------------------------------------------

def save_calibrator(
    calibrator: SegmentedTransportCalibrator,
    path: str | Path,
) -> None:
    """Save calibrator state to a JSON + npz bundle.

    Parameters
    ----------
    calibrator : SegmentedTransportCalibrator
        The calibrator to save.
    path : str or Path
        Directory path or .zip file path. If the path ends with '.zip',
        a zip bundle is created. Otherwise, a directory with meta.json
        and arrays.npz is created.
    """
    import io
    import zipfile

    path = Path(path)
    state = calibrator._get_state()

    # Separate arrays from metadata
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    meta = {}
    for k, v in state.items():
        if isinstance(v, (list,)) and len(v) > 10:
            arrays[k] = np.array(v)
        else:
            meta[k] = v

    meta_bytes = json.dumps(meta, indent=2, default=str).encode("utf-8")

    array_bytes = b""
    if arrays:
        buf = io.BytesIO()
        np.savez_compressed(cast(Any, buf), **cast(Any, arrays))
        array_bytes = buf.getvalue()

    if path.suffix == ".zip":
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", meta_bytes)
            if array_bytes:
                zf.writestr("arrays.npz", array_bytes)
    else:
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "meta.json", "wb") as f:
            f.write(meta_bytes)
        if array_bytes:
            with open(path / "arrays.npz", "wb") as f:
                f.write(array_bytes)


def load_calibrator(
    path: str | Path,
    detector_factory: Optional[Callable[..., SegmentDetector]] = None,
    step_schedule: Optional[Callable[[float], float]] = None,
) -> SegmentedTransportCalibrator:
    """Load calibrator state from a saved bundle.

    Parameters
    ----------
    path : str or Path
        Directory containing meta.json and arrays.npz, or a .zip bundle.
    detector_factory : callable, optional
        Factory accepting saved detector state dict and returning a
        reconstructed detector. If None, reconstructs from saved type.
    step_schedule : callable, optional
        Step schedule to use for future updates. Exact continuation with
        a non-default saved schedule requires passing the same callable.

    Returns
    -------
    SegmentedTransportCalibrator
        Restored calibrator ready to continue online.
    """
    import io
    import zipfile
    from tsconformal.detectors import CUSUMNormDetector, PageHinkleyDetector

    _DETECTOR_REGISTRY = {
        "CUSUMNormDetector": CUSUMNormDetector,
        "PageHinkleyDetector": PageHinkleyDetector,
    }

    path = Path(path)

    # Load from zip or directory
    if path.suffix == ".zip" or path.is_file():
        with zipfile.ZipFile(path, "r") as zf:
            meta = json.loads(zf.read("meta.json"))
            arrays = {}
            if "arrays.npz" in zf.namelist():
                buf = io.BytesIO(zf.read("arrays.npz"))
                with np.load(buf) as data:
                    arrays = dict(data)
    else:
        with open(path / "meta.json") as f:
            meta = json.load(f)
        arrays = {}
        npz_path = path / "arrays.npz"
        if npz_path.exists():
            with np.load(npz_path) as data:
                arrays = dict(data)

    # Merge
    state = {**meta}
    for k, v in arrays.items():
        state[k] = v.tolist()

    # Validate schema
    _validate_state_schema_version(state.get("schema_version"))

    # Reconstruct detector
    det_state = state.get("detector_state", {})
    det_type_name = state.get("detector_type", "CUSUMNormDetector")

    if detector_factory is not None:
        detector = detector_factory(det_state)
    elif det_type_name in _DETECTOR_REGISTRY:
        det_cls = _DETECTOR_REGISTRY[det_type_name]
        if det_cls is CUSUMNormDetector:
            detector = CUSUMNormDetector(
                kappa=det_state.get("kappa", 0.02),
                threshold=det_state.get("threshold", 0.20),
            )
            detector._S = det_state.get("S", 0.0)
            detector._t = det_state.get("t", 0)
        elif det_cls is PageHinkleyDetector:
            detector = PageHinkleyDetector(
                delta=det_state.get("delta", 0.01),
                threshold=det_state.get("threshold", 0.50),
            )
            detector._m = det_state.get("m", 0.0)
            detector._M = det_state.get("M", 0.0)
            detector._t = det_state.get("t", 0)
        else:
            detector = cast(SegmentDetector, det_cls())
    else:
        raise ValueError(
            f"Cannot reconstruct detector type '{det_type_name}'. "
            f"Provide a detector_factory argument."
        )

    # Resolve step schedule
    saved_schedule_id = state.get("step_schedule_id", _DEFAULT_STEP_SCHEDULE_ID)
    if step_schedule is None:
        if saved_schedule_id == _DEFAULT_STEP_SCHEDULE_ID:
            step_schedule = _default_step
        else:
            raise ValueError(
                "Saved calibrator uses a non-default step_schedule. "
                "Pass the same step_schedule to load_calibrator for exact continuation."
            )
    else:
        loaded_schedule_id = _resolve_step_schedule_id(step_schedule)
        if (
            saved_schedule_id not in {_CUSTOM_STEP_SCHEDULE_ID, loaded_schedule_id}
            or (
                saved_schedule_id == _DEFAULT_STEP_SCHEDULE_ID
                and loaded_schedule_id != _DEFAULT_STEP_SCHEDULE_ID
            )
        ):
            warnings.warn(
                f"Saved step_schedule was '{saved_schedule_id}' but loading "
                f"with '{loaded_schedule_id}'. Exact continuation may differ.",
                RuntimeWarning,
                stacklevel=2,
            )

    # Reconstruct calibrator
    cal = SegmentedTransportCalibrator(
        grid_size=state["grid_size"],
        rho=state["rho"],
        n_eff_min=state["n_eff_min"],
        step_schedule=step_schedule,
        detector=detector,
        cooldown=state["cooldown"],
        confirm=state["confirm"],
        projection=state.get("projection", "isotonic_l2"),
        interpolation=state.get("interpolation", "piecewise_linear"),
        fallback=state.get("fallback", "identity"),
        warm_start=state.get("warm_start", "identity"),
    )
    cal._set_state(state)

    return cal
