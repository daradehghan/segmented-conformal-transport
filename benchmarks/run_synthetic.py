"""Synthetic benchmark runner for Paper 1.

Runs SCT vs ACI vs split-CP on representative DGP configurations,
computes all spec-required metrics, and produces results tables.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from tsconformal._benchmark_defaults import (
    DEFAULT_PIT_SEED,
    build_sct_calibrator,
    make_pit_rng,
)
from tsconformal.metrics import (
    gridwise_calibration_error,
    marginal_coverage,
    mean_interval_width,
)
from tsconformal.diagnostics import (
    pit_uniformity_tests,
    warm_start_occupancy,
)
from tsconformal.examples.synthetic_piecewise_stationary import GaussianForecastCDF

from benchmarks.synthetic.dgp import (
    generate_family_A,
    generate_family_B,
    generate_family_C,
    generate_family_D,
    generate_family_F,
    generate_family_G,
    SyntheticStream,
)
from benchmarks.baselines import ACIBaseline, SplitCPBaseline


# -----------------------------------------------------------------------
# SCT runner
# -----------------------------------------------------------------------

@dataclass
class MethodResult:
    """Results for a single method on a single stream."""
    method: str
    E_r: float = np.nan
    coverage_90: float = np.nan
    coverage_80: float = np.nan
    width_90: float = np.nan
    width_80: float = np.nan
    pit_ks_p: float = np.nan
    n_resets: int = 0
    warm_start_occ: float = np.nan
    wall_seconds: float = 0.0


def regimewise_calibration_error(
    y: np.ndarray,
    calibrated_quantiles: np.ndarray,
    regime_ids: np.ndarray,
    J_eval: int = 99,
) -> float:
    """Return the length-weighted mean of true regime-wise calibration errors."""
    y = np.asarray(y)
    Q = np.asarray(calibrated_quantiles)
    regimes = np.asarray(regime_ids)

    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if Q.ndim != 2 or Q.shape[0] != len(y):
        raise ValueError("calibrated_quantiles must have shape (n, J_eval)")
    if regimes.shape != y.shape:
        raise ValueError("regime_ids must have the same length as y")

    errors = []
    weights = []
    for regime_id in np.unique(regimes):
        mask = regimes == regime_id
        if not np.any(mask):
            continue
        errors.append(
            gridwise_calibration_error(y[mask], Q[mask], J_eval)
        )
        weights.append(int(mask.sum()))

    return float(np.average(errors, weights=weights))


def run_sct(
    stream: SyntheticStream,
    alpha: float = 0.10,
    pit_seed: int = DEFAULT_PIT_SEED,
) -> MethodResult:
    """Run SCT on a synthetic stream."""
    t0 = time.time()
    cal = build_sct_calibrator(
        grid_size=49,
        rho=0.99,
        n_eff_min=30,
        cooldown=100,
        confirm=3,
    )
    pit_rng = make_pit_rng(pit_seed)

    T = len(stream.y)
    J_eval = 99
    eval_grid = np.array([j / (J_eval + 1) for j in range(1, J_eval + 1)])

    cal_quantiles = np.zeros((T, J_eval))
    lowers_90 = np.zeros(T)
    uppers_90 = np.zeros(T)
    lowers_80 = np.zeros(T)
    uppers_80 = np.zeros(T)
    pits = np.zeros(T)
    warmup_flags = np.zeros(T, dtype=bool)
    ws_weights = np.zeros(T)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(T):
            cdf = GaussianForecastCDF(
                mu=stream.forecast_mu[t],
                sigma=max(stream.forecast_sigma[t], 1e-8),
            )
            cal_cdf = cal.predict_cdf(cdf)

            # Record calibrated quantiles for E_r
            for j, u in enumerate(eval_grid):
                cal_quantiles[t, j] = cal_cdf.ppf(u)

            lowers_90[t] = cal_cdf.ppf(0.05)
            uppers_90[t] = cal_cdf.ppf(0.95)
            lowers_80[t] = cal_cdf.ppf(0.10)
            uppers_80[t] = cal_cdf.ppf(0.90)
            pits[t] = cal_cdf.cdf(stream.y[t])
            warmup_flags[t] = cal.in_warmup
            ws_weights[t] = cal.warm_start_weight

            cal.update(stream.y[t], cdf, rng=pit_rng)

    # Synthetic streams have oracle regime labels, so report a stream-level
    # summary of the true regime-wise E_r values rather than a whole-stream score.
    E_r = regimewise_calibration_error(
        stream.y,
        cal_quantiles,
        stream.regime_ids,
        J_eval,
    )
    cov_90 = marginal_coverage(stream.y, lowers_90, uppers_90)
    cov_80 = marginal_coverage(stream.y, lowers_80, uppers_80)
    w_90 = mean_interval_width(lowers_90, uppers_90)
    w_80 = mean_interval_width(lowers_80, uppers_80)
    pit_diag = pit_uniformity_tests(pits)
    ws_occ = warm_start_occupancy(warmup_flags, ws_weights)

    return MethodResult(
        method="SCT",
        E_r=E_r, coverage_90=cov_90, coverage_80=cov_80,
        width_90=w_90, width_80=w_80,
        pit_ks_p=pit_diag.ks_pvalue,
        n_resets=cal.num_resets,
        warm_start_occ=ws_occ,
        wall_seconds=time.time() - t0,
    )


def run_aci(stream: SyntheticStream, alpha: float = 0.10) -> MethodResult:
    """Run ACI baseline on a synthetic stream."""
    t0 = time.time()
    aci_90 = ACIBaseline(alpha=0.10, gamma=0.005)
    aci_80 = ACIBaseline(alpha=0.20, gamma=0.005)

    T = len(stream.y)
    lowers_90 = np.zeros(T)
    uppers_90 = np.zeros(T)
    lowers_80 = np.zeros(T)
    uppers_80 = np.zeros(T)

    for t in range(T):
        y_hat = stream.forecast_mu[t]  # point forecast = miscalibrated mean
        l90, u90 = aci_90.predict_interval(y_hat)
        l80, u80 = aci_80.predict_interval(y_hat)
        lowers_90[t] = l90
        uppers_90[t] = u90
        lowers_80[t] = l80
        uppers_80[t] = u80
        aci_90.update(stream.y[t], y_hat)
        aci_80.update(stream.y[t], y_hat)

    cov_90 = marginal_coverage(stream.y, lowers_90, uppers_90)
    cov_80 = marginal_coverage(stream.y, lowers_80, uppers_80)
    w_90 = mean_interval_width(lowers_90, uppers_90)
    w_80 = mean_interval_width(lowers_80, uppers_80)

    return MethodResult(
        method="ACI",
        E_r=np.nan,  # interval-only → E_r = N/A
        coverage_90=cov_90, coverage_80=cov_80,
        width_90=w_90, width_80=w_80,
        wall_seconds=time.time() - t0,
    )


def run_split_cp(stream: SyntheticStream, alpha: float = 0.10,
                 n_cal: int = 300) -> MethodResult:
    """Run split-CP baseline on a synthetic stream."""
    t0 = time.time()
    T = len(stream.y)
    if T <= n_cal + 10:
        return MethodResult(method="split-CP")

    # Calibration block: first n_cal points
    y_cal = stream.y[:n_cal]
    yhat_cal = stream.forecast_mu[:n_cal]

    cp_90 = SplitCPBaseline(alpha=0.10)
    cp_80 = SplitCPBaseline(alpha=0.20)
    cp_90.calibrate(y_cal, yhat_cal)
    cp_80.calibrate(y_cal, yhat_cal)

    # Test on remaining
    test_start = n_cal
    T_test = T - test_start
    lowers_90 = np.zeros(T_test)
    uppers_90 = np.zeros(T_test)
    lowers_80 = np.zeros(T_test)
    uppers_80 = np.zeros(T_test)

    for t in range(T_test):
        y_hat = stream.forecast_mu[test_start + t]
        l90, u90 = cp_90.predict_interval(y_hat)
        l80, u80 = cp_80.predict_interval(y_hat)
        lowers_90[t] = l90
        uppers_90[t] = u90
        lowers_80[t] = l80
        uppers_80[t] = u80

    y_test = stream.y[test_start:]
    cov_90 = marginal_coverage(y_test, lowers_90, uppers_90)
    cov_80 = marginal_coverage(y_test, lowers_80, uppers_80)
    w_90 = mean_interval_width(lowers_90, uppers_90)
    w_80 = mean_interval_width(lowers_80, uppers_80)

    return MethodResult(
        method="split-CP",
        E_r=np.nan,  # interval-only → E_r = N/A
        coverage_90=cov_90, coverage_80=cov_80,
        width_90=w_90, width_80=w_80,
        wall_seconds=time.time() - t0,
    )


# -----------------------------------------------------------------------
# Scenario configurations (representative subset)
# -----------------------------------------------------------------------

def get_scenarios() -> List[dict]:
    """Return representative scenario configurations."""
    scenarios = []

    # Family A: piecewise-stationary
    for phi in [0.0, 0.5]:
        scenarios.append({
            "name": f"A: R=3, L=500, phi={phi}",
            "gen": lambda s, p=phi: generate_family_A(
                R_reg=3, L=500, phi=p, seed=s),
        })

    # Family B: bounded drift
    for lam in [0.0, 0.2, 0.4]:
        scenarios.append({
            "name": f"B: L=1000, λ={lam}",
            "gen": lambda seed, drift=lam: generate_family_B(
                L=1000, lambda_r=drift, phi=0.5, seed=seed),
        })

    # Family C: abrupt change
    for db in [0.5, 1.0]:
        scenarios.append({
            "name": f"C: L=500, Δb={db}",
            "gen": lambda s, d=db: generate_family_C(
                L=500, Delta_b=d, Delta_s=0.3, phi=0.5, seed=s),
        })

    # Family D: oscillating short
    for ls in [25, 100]:
        scenarios.append({
            "name": f"D: L_short={ls}",
            "gen": lambda seed, short_length=ls: generate_family_D(
                L_short=short_length, phi=0.5, T_scored=3000, seed=seed),
        })

    # Family F: heavy tails
    for nu in [3, 5]:
        scenarios.append({
            "name": f"F: ν={nu}",
            "gen": lambda s, n=nu: generate_family_F(
                nu=n, s=1.2, phi=0.5, seed=s),
        })

    # Family G: drifting forecaster
    for gamma in [0.0, 0.4, 0.8]:
        scenarios.append({
            "name": f"G: γ={gamma}",
            "gen": lambda s, g=gamma: generate_family_G(
                gamma=g, phi=0.5, T_scored=3000, seed=s),
        })

    return scenarios


# -----------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------

def run_benchmark(
    n_replicates: int = 10,
    seed_base: int = DEFAULT_PIT_SEED,
    pit_seed: int = DEFAULT_PIT_SEED,
):
    """Run the full synthetic benchmark."""
    scenarios = get_scenarios()

    print("=" * 80)
    print(f"SYNTHETIC BENCHMARK — {len(scenarios)} scenarios × "
          f"{n_replicates} replicates × 3 methods")
    print("=" * 80)

    all_results = []

    for sc in scenarios:
        sct_results = []
        aci_results = []
        cp_results = []

        for rep in range(n_replicates):
            seed = seed_base + rep
            stream = sc["gen"](seed)

            sct_r = run_sct(stream, pit_seed=pit_seed)
            aci_r = run_aci(stream)
            cp_r = run_split_cp(stream)

            sct_results.append(sct_r)
            aci_results.append(aci_r)
            cp_results.append(cp_r)

        # Aggregate: median across replicates
        def med(results, attr):
            vals = [getattr(r, attr) for r in results]
            vals = [v for v in vals if np.isfinite(v)]
            return np.median(vals) if vals else np.nan

        print(f"\n--- {sc['name']} ---")
        print(f"{'Method':<10} {'E_r':>8} {'Cov90':>8} {'Cov80':>8} "
              f"{'W90':>8} {'W80':>8} {'PIT_KS':>8} {'Resets':>7} "
              f"{'WS_occ':>7} {'Time':>6}")
        print("-" * 90)

        for label, results in [("SCT", sct_results), ("ACI", aci_results),
                                ("split-CP", cp_results)]:
            e_r = med(results, "E_r")
            c90 = med(results, "coverage_90")
            c80 = med(results, "coverage_80")
            w90 = med(results, "width_90")
            w80 = med(results, "width_80")
            ks = med(results, "pit_ks_p")
            resets = int(np.median([r.n_resets for r in results]))
            ws = med(results, "warm_start_occ")
            t_s = np.mean([r.wall_seconds for r in results])

            e_r_str = f"{e_r:.4f}" if np.isfinite(e_r) else "N/A"
            ks_str = f"{ks:.4f}" if np.isfinite(ks) else "N/A"
            ws_str = f"{ws:.3f}" if np.isfinite(ws) else "N/A"

            print(f"{label:<10} {e_r_str:>8} {c90:>8.3f} {c80:>8.3f} "
                  f"{w90:>8.3f} {w80:>8.3f} {ks_str:>8} {resets:>7} "
                  f"{ws_str:>7} {t_s:>5.1f}s")

            all_results.append({
                "scenario": sc["name"], "method": label,
                "E_r": e_r, "coverage_90": c90, "coverage_80": c80,
                "width_90": w90, "width_80": w80,
                "pit_ks_p": ks, "resets": resets,
                "warm_start_occ": ws, "wall_seconds": t_s,
            })

    return all_results


def build_synthetic_snapshot(
    results: List[dict],
    n_replicates: int,
    seed_base: int,
    pit_seed: int,
) -> dict:
    """Wrap aggregated synthetic benchmark rows in a structured snapshot."""
    return {
        "script": "benchmarks/run_synthetic.py",
        "n_replicates": int(n_replicates),
        "seed_base": int(seed_base),
        "pit_seed": int(pit_seed),
        "n_rows": int(len(results)),
        "results": results,
    }


def save_synthetic_snapshot(
    path: Path | str,
    results: List[dict],
    n_replicates: int,
    seed_base: int,
    pit_seed: int,
) -> None:
    """Save a structured synthetic benchmark snapshot as JSON."""
    snapshot = build_synthetic_snapshot(
        results=results,
        n_replicates=n_replicates,
        seed_base=seed_base,
        pit_seed=pit_seed,
    )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(snapshot, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the synthetic SCT benchmark")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_PIT_SEED)
    parser.add_argument("--pit-seed", type=int, default=DEFAULT_PIT_SEED)
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path for a structured synthetic benchmark snapshot.",
    )
    args = parser.parse_args()

    results = run_benchmark(
        n_replicates=args.n_replicates,
        seed_base=args.seed_base,
        pit_seed=args.pit_seed,
    )
    print(f"\nTotal results: {len(results)}")
    if args.output_json:
        save_synthetic_snapshot(
            path=args.output_json,
            results=results,
            n_replicates=args.n_replicates,
            seed_base=args.seed_base,
            pit_seed=args.pit_seed,
        )
        print(f"Saved structured JSON snapshot to {args.output_json}")
