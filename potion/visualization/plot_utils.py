"""Reusable helpers for plotting learning curves."""

from pathlib import Path

import matplot2tikz
import numpy as np


DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 42


def bootstrap_mean_confidence_interval(
    performance,
    confidence=DEFAULT_CONFIDENCE,
    n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
    rng=None,
):
    """Compute a pointwise percentile interval for the mean across curves."""
    performance = np.asarray(performance)
    if performance.ndim != 2 or performance.shape[0] < 2:
        raise ValueError("At least two curves are required for bootstrapping")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    if rng is None:
        rng = np.random.default_rng(DEFAULT_BOOTSTRAP_SEED)

    n_curves = performance.shape[0]
    sample_indices = rng.integers(
        0, n_curves, size=(n_bootstrap, n_curves)
    )
    bootstrap_means = performance[sample_indices].mean(axis=1)
    tail_probability = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail_probability, 1.0 - tail_probability],
        axis=0,
    )
    return performance.mean(axis=0), lower, upper


def plot_bootstrap_mean_curve(
    axis,
    x,
    performance,
    *,
    label,
    color,
    marker,
    confidence=DEFAULT_CONFIDENCE,
    n_bootstrap=DEFAULT_BOOTSTRAP_SAMPLES,
    rng=None,
    marker_count=10,
):
    """Plot a mean curve and its pointwise bootstrap confidence band."""
    x = np.asarray(x)
    performance = np.asarray(performance)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if performance.ndim != 2 or performance.shape[1] != len(x):
        raise ValueError("performance must have shape (curves, len(x))")
    if marker_count < 1:
        raise ValueError("marker_count must be at least 1")

    mean, lower, upper = bootstrap_mean_confidence_interval(
        performance,
        confidence=confidence,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )
    plot_mean_curve(
        axis,
        x,
        mean,
        lower,
        upper,
        label=label,
        color=color,
        marker=marker,
        marker_count=marker_count,
    )
    return mean, lower, upper


def plot_mean_curve(
    axis,
    x,
    mean,
    lower,
    upper,
    *,
    label,
    color,
    marker,
    marker_count=10,
):
    """Plot a precomputed mean curve and confidence band."""
    x = np.asarray(x)
    mean = np.asarray(mean)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if any(values.shape != x.shape for values in (mean, lower, upper)):
        raise ValueError("mean and confidence bounds must have the same shape as x")
    if marker_count < 1:
        raise ValueError("marker_count must be at least 1")

    marker_interval = max(1, len(x) // marker_count)
    axis.plot(
        x,
        mean,
        color=color,
        marker=marker,
        markevery=marker_interval,
        linewidth=2,
        markersize=6,
        label=label,
    )
    axis.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=0.2,
        linewidth=0,
    )


def style_learning_curve_axis(
    axis,
    *,
    x_label="Training trajectories",
    y_label="Test performance",
):
    """Apply the common labels and legend layout for a learning-curve plot."""
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.legend()
    axis.margins(x=0)


def save_png_and_tikz(
    figure,
    output_path,
    *,
    dpi=200,
    tikz_width=r"\figurewidth",
    tikz_height=r"\figureheight",
):
    """Save a Matplotlib figure as matching PNG and PGFPlots/TikZ files."""
    output_path = Path(output_path)
    output_stem = output_path.with_suffix("") if output_path.suffix else output_path
    png_path = output_stem.with_suffix(".png")
    tikz_path = output_stem.with_suffix(".tex")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    matplot2tikz.save(
        tikz_path,
        figure=figure,
        axis_width=tikz_width,
        axis_height=tikz_height,
    )
    return png_path, tikz_path
