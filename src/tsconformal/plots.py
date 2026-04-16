"""Plotting helpers for benchmark visuals.

Requires optional dependency: matplotlib.
"""

from __future__ import annotations

import numpy as np


def _get_plt():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError("matplotlib required for plots")


def pit_reliability_diagram(pits, n_bins=20, ax=None):
    """PIT reliability diagram."""
    plt = _get_plt()
    pits = np.asarray(pits)
    if ax is None:
        _, ax = plt.subplots()
    edges = np.linspace(0, 1, n_bins + 1)
    observed = np.histogram(pits, bins=edges)[0] / len(pits)
    expected = 1.0 / n_bins
    centers = (edges[:-1] + edges[1:]) / 2
    ax.bar(centers, observed, width=1.0 / n_bins, alpha=0.7, label="Observed")
    ax.axhline(expected, color="red", linestyle="--", label="Uniform")
    ax.set_xlabel("PIT bin")
    ax.set_ylabel("Density")
    ax.legend()
    return ax


def coverage_over_time(rolling_cov, target, ax=None):
    """Plot rolling coverage over time."""
    plt = _get_plt()
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(rolling_cov, label="Rolling coverage")
    ax.axhline(target, color="red", linestyle="--", label=f"Target={target:.2f}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Coverage")
    ax.legend()
    return ax


def width_vs_coverage(
    method_names: list,
    coverages: list,
    widths: list,
    target_coverage: float = 0.90,
    ax=None,
):
    """Width-vs-coverage scatter/tradeoff plot across methods.

    Parameters
    ----------
    method_names : list of str
    coverages : list of float
    widths : list of float
    target_coverage : float
    """
    plt = _get_plt()
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(coverages, widths, s=80, zorder=3)
    for i, name in enumerate(method_names):
        ax.annotate(name, (coverages[i], widths[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.axvline(target_coverage, color="red", linestyle="--", alpha=0.5,
               label=f"Target={target_coverage:.2f}")
    ax.set_xlabel("Empirical coverage")
    ax.set_ylabel("Mean interval width")
    ax.set_title("Width vs Coverage")
    ax.legend()
    return ax


def sensitivity_heatmap(
    thresholds: list,
    cooldowns: list,
    metric_matrix: np.ndarray,
    metric_name: str = "E_r",
    ax=None,
):
    """Detector sensitivity heatmap.

    Parameters
    ----------
    thresholds : list of float
        Detector threshold values (rows).
    cooldowns : list of int
        Cooldown values (columns).
    metric_matrix : ndarray of shape (len(thresholds), len(cooldowns))
    metric_name : str
    """
    plt = _get_plt()
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(metric_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(cooldowns)))
    ax.set_xticklabels([str(c) for c in cooldowns])
    ax.set_yticks(range(len(thresholds)))
    ax.set_yticklabels([f"{h:.2f}" for h in thresholds])
    ax.set_xlabel("Cooldown")
    ax.set_ylabel("Threshold")
    ax.set_title(f"Detector Sensitivity: {metric_name}")
    plt.colorbar(im, ax=ax)
    return ax


def lag_vs_width(
    method_names: list,
    adaptation_lags: list,
    mean_widths_adapt: list,
    ax=None,
):
    """Adaptation lag vs mean width during adaptation window.

    Parameters
    ----------
    method_names : list of str
    adaptation_lags : list of float or None
        None values are plotted at the right edge as censored.
    mean_widths_adapt : list of float
    """
    plt = _get_plt()
    if ax is None:
        _, ax = plt.subplots()
    max_lag = max(l for l in adaptation_lags if l is not None) * 1.2 if any(
        l is not None for l in adaptation_lags) else 100
    for i, name in enumerate(method_names):
        lag = adaptation_lags[i] if adaptation_lags[i] is not None else max_lag
        marker = "o" if adaptation_lags[i] is not None else "x"
        ax.scatter(lag, mean_widths_adapt[i], s=80, marker=marker, zorder=3)
        ax.annotate(name, (lag, mean_widths_adapt[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Adaptation lag (censored at right edge)")
    ax.set_ylabel("Mean width during adaptation")
    ax.set_title("Lag vs Width Tradeoff")
    return ax
