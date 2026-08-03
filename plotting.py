"""Generate figures from processed self-potential output tables.

This module reads processed CSV products defined by a TOML configuration file
and generates a standard set of figures for self-potential, integrated
electric potential, temperature, and conductivity. Missing CSV inputs are
treated as optional: the script prints a terminal warning and skips any plots
that depend on unavailable files.

Outlier handling
----------------
Real surveys contain electrode dropouts and contact spikes that are orders of
magnitude larger than the signal of interest. Left alone, Matplotlib's
autoscaling stretches the y-axis to contain them and flattens everything else.

Two exclusion tables at the top of this module control which samples are
allowed to influence the y-axis limits:

* `EXCLUDE_X_RANGES` - windows along the x-axis (segment distance in km, or
  sample index, depending on the plot) whose samples are ignored when the
  limits are computed.
* `EXCLUDE_Y_RANGES` - value windows; any y value falling inside one of them is
  ignored. Use `float("-inf")` or `float("inf")` for one-sided cuts.

Excluded samples are still plotted. Matplotlib simply clips them at the axis
edge, so nothing is silently deleted from the figure. Excluded x windows are
shaded so the exclusion stays visible when the figure is reviewed later.

When a figure needs a fixed frame instead - for a report, or to compare two
surveys on identical axes - `MANUAL_Y_LIMITS` and `MANUAL_X_LIMITS` set hard
bounds per axis and bypass the automatic calculation entirely.

The main entry point is `main()`, which parses a config path from the command
line and writes the figures into the configured figures directory.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib
from matplotlib.axes import Axes
from matplotlib.figure import Figure

ConfigDict = dict[str, Any]
Range = tuple[float, float]


# =============================================================================
# AUTOSCALE SETTINGS
# =============================================================================

# X-axis windows to ignore when computing y-limits, keyed by axis.
#
# Units follow whatever that axis plots:
#   sp_segments                -> segment distance, km
#   interpretation_upstream    -> sample index
#   interpretation_downstream  -> sample index
#   integrated_*               -> sample index
#   temperature, conductivity  -> segment distance, km
#
# Example - drop a bad electrode stretch and a launch transient:
#   "sp_segments": [(0.00, 0.02), (0.61, 0.68)],
EXCLUDE_X_RANGES: dict[str, list[Range]] = {
    "sp_segments": [],
    "interpretation_upstream": [],
    "interpretation_downstream": [],
    "integrated_V_full": [],
    "integrated_VL_lowfreq": [],
    "integrated_VH_highfreq": [],
    "integrated_VN_noise": [],
    "temperature": [],
    "conductivity": [],
}

# Value windows to ignore when computing y-limits, keyed by the same axis names.
#
# Example - ignore anything beyond +/- 500 mV:
#   "sp_segments": [(-float("inf"), -500.0), (500.0, float("inf"))],
EXCLUDE_Y_RANGES: dict[str, list[Range]] = {
    "sp_segments": [],
    "interpretation_upstream": [],
    "interpretation_downstream": [],
    "integrated_V_full": [],
    "integrated_VL_lowfreq": [],
    "integrated_VH_highfreq": [],
    "integrated_VN_noise": [],
    "temperature": [],
    "conductivity": [],
}

# Hard y-limits, keyed by the same axis names. An entry that is not None wins
# outright: the exclusion tables and percentile clipping are skipped for that
# axis. Either bound may be None to pin one end and autoscale the other.
#
# Examples:
#   "sp_segments": (-40.0, 60.0),     both ends fixed
#   "temperature": (None, 3.0),       cap the top, autoscale the bottom
MANUAL_Y_LIMITS: dict[str, tuple[float | None, float | None] | None] = {
    "sp_segments": (-4, 8),
    "interpretation_upstream": (-4, 8),
    "interpretation_downstream": None,
    "integrated_V_full": None,
    "integrated_VL_lowfreq": None,
    "integrated_VH_highfreq": None,
    "integrated_VN_noise": None,
    "temperature": None,
    "conductivity": None,
}

# Hard x-limits, same convention. Useful for zooming into a reach without
# touching the processed tables. Note that this only changes the view; it does
# not remove off-screen samples from the y-limit calculation, so pair it with
# EXCLUDE_X_RANGES if you want the y-axis to follow the zoom.
MANUAL_X_LIMITS: dict[str, tuple[float | None, float | None] | None] = {
    "sp_segments": None,
    "interpretation_upstream": None,
    "interpretation_downstream": None,
    "integrated_V_full": None,
    "integrated_VL_lowfreq": None,
    "integrated_VH_highfreq": None,
    "integrated_VN_noise": None,
    "temperature": None,
    "conductivity": None,
}

# Headroom added above and below the surviving data range.
Y_PAD_FRAC = 0.05

# Optional automatic outlier rejection applied after the explicit exclusions.
# Set to a (low, high) percentile pair such as (1.0, 99.0) to clip without
# hand-listing ranges, or None to use the full surviving min/max.
AUTOSCALE_PERCENTILES: tuple[float, float] | None = None

# Shade excluded x windows so the exclusion is visible in the saved figure.
SHADE_EXCLUDED_X = True


# =============================================================================
# AUTOSCALE HELPER
# =============================================================================

class AxisScaler:
    """Accumulate plotted series and set y-limits from non-excluded samples.

    Feed every series drawn on an axis through `add()`, then call `apply()`
    once after all plotting on that axis is finished.

    Attributes:
        key: Axis name used to look up entries in the exclusion tables.
    """

    def __init__(self, key: str) -> None:
        """Initialise a scaler for one axis.

        Args:
            key: Axis name; must match a key in the exclusion tables to have
                any effect. Unknown keys simply apply no exclusions.
        """
        self.key = key
        self.x_excl: list[Range] = list(EXCLUDE_X_RANGES.get(key, ()))
        self.y_excl: list[Range] = list(EXCLUDE_Y_RANGES.get(key, ()))
        self.y_manual = MANUAL_Y_LIMITS.get(key)
        self.x_manual = MANUAL_X_LIMITS.get(key)
        self._kept: list[np.ndarray] = []
        self._n_finite = 0
        self._n_dropped = 0

    def add(self, x: Any, y: Any) -> None:
        """Register one plotted series with the scaler.

        Args:
            x: X values of the series, in that axis's own units.
            y: Y values of the series.

        Raises:
            ValueError: If `x` and `y` have different lengths.
        """
        xv = np.asarray(x, dtype=float)
        yv = np.asarray(y, dtype=float)
        if xv.size != yv.size:
            raise ValueError(
                f"AxisScaler[{self.key}]: x has {xv.size} points, y has {yv.size}"
            )

        keep = np.isfinite(yv)
        n_finite = int(keep.sum())

        for lo, hi in self.x_excl:
            keep &= ~((xv >= lo) & (xv <= hi))
        for lo, hi in self.y_excl:
            keep &= ~((yv >= lo) & (yv <= hi))

        self._n_finite += n_finite
        self._n_dropped += n_finite - int(keep.sum())
        if keep.any():
            self._kept.append(yv[keep])

    def apply(self, ax: Axes) -> None:
        """Shade excluded windows and set the limits on `ax`.

        Manual limits take priority. When only one manual y bound is given,
        the other end is autoscaled from the surviving samples first.

        Args:
            ax: Axis to rescale. If every sample was excluded, or no series
                were registered, Matplotlib's own autoscaling is left in place.
        """
        if self.x_excl and SHADE_EXCLUDED_X:
            xlim = ax.get_xlim()
            for lo, hi in self.x_excl:
                ax.axvspan(lo, hi, color="0.80", alpha=0.45, lw=0, zorder=0)
            ax.set_xlim(xlim)

        if self.x_manual is not None:
            x_lo, x_hi = self.x_manual
            ax.set_xlim(left=x_lo, right=x_hi)
            print(f"[INFO] '{self.key}': manual x-limits {x_lo} to {x_hi}")

        if self.y_manual is not None:
            y_lo, y_hi = self.y_manual
            if y_lo is None or y_hi is None:
                # Autoscale the open end from the surviving samples first, so
                # the pinned end does not drag the free end along with it.
                self._autoscale_y(ax)
            ax.set_ylim(bottom=y_lo, top=y_hi)
            print(f"[INFO] '{self.key}': manual y-limits {y_lo} to {y_hi}")
            return

        self._autoscale_y(ax)

    def _autoscale_y(self, ax: Axes) -> None:
        """Set y-limits from the samples that survived the exclusion tables.

        Args:
            ax: Axis to rescale.
        """
        if not self._kept:
            if self.x_excl or self.y_excl:
                print(
                    f"[WARN] '{self.key}': exclusions removed every sample; "
                    "leaving automatic scaling in place."
                )
            return

        vals = np.concatenate(self._kept)

        if AUTOSCALE_PERCENTILES is not None:
            lo_p, hi_p = AUTOSCALE_PERCENTILES
            lo, hi = (float(v) for v in np.percentile(vals, [lo_p, hi_p]))
        else:
            lo, hi = float(vals.min()), float(vals.max())

        if not (math.isfinite(lo) and math.isfinite(hi)):
            return

        span = hi - lo
        if span <= 0.0:
            span = max(abs(hi), 1.0) * 0.1
        pad = span * Y_PAD_FRAC
        ax.set_ylim(lo - pad, hi + pad)

        if self._n_dropped:
            print(
                f"[INFO] '{self.key}': {self._n_dropped} of {self._n_finite} "
                f"samples excluded from y-scaling; "
                f"limits {lo - pad:.4g} to {hi + pad:.4g}"
            )


# =============================================================================
# IO
# =============================================================================

def load_config(config_path: str | Path = "config.toml") -> ConfigDict:
    """Load TOML configuration from disk.

    Args:
        config_path: Path to the TOML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        tomllib.TOMLDecodeError: If the TOML file cannot be parsed.
    """
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        return tomllib.load(f)


def ensure_dir(path: Path) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(base: Path, name: str) -> pd.DataFrame | None:
    """Read a CSV file if it exists.

    Args:
        base: Base directory containing the file.
        name: File name relative to `base`.

    Returns:
        The loaded CSV as a pandas DataFrame if the file exists; otherwise
        `None`.

    Notes:
        Missing files are treated as optional. A warning is printed to the
        terminal and the corresponding plots should be skipped by the caller.
    """
    path = base / name
    if not path.exists():
        print(f"[WARN] Missing input file, skipping plots that depend on it: {path}")
        return None
    return pd.read_csv(path)


def savefig(fig: Figure, path: Path) -> None:
    """Save and close a Matplotlib figure.

    Args:
        fig: Figure to save.
        path: Output image path.
    """
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURES
# =============================================================================

def plot_gradient_sp_segments(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot drift-corrected gradient self-potential by segment.

    Args:
        df: DataFrame containing the processed gradient SP table.
        figures_dir: Output directory for figure files.

    Notes:
        Y-limits honour the `sp_segments` entries in the exclusion tables.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors: dict[int, str] = {1: "k", 2: "b", 3: "c", 4: "m"}
    labels: dict[int, str] = {
        1: "Segment 1",
        2: "Segment 2",
        3: "Segment 3",
        4: "Segment 4",
    }

    scaler = AxisScaler("sp_segments")

    for seg in sorted(df["segment_id"].unique()):
        d = df[df["segment_id"] == seg].copy()
        ax.plot(
            d["segment_distance_km"],
            d["drift_corrected_SP_mV"],
            color=colors.get(int(seg), None),
            lw=1.2,
            label=labels.get(int(seg), f"Segment {seg}"),
        )
        scaler.add(d["segment_distance_km"], d["drift_corrected_SP_mV"])

    ax.set_xlabel("Segment Distance (km)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Drift-Corrected Gradient SP by Segment")
    ax.legend()
    ax.minorticks_on()
    ax.grid(alpha=0.2)
    scaler.apply(ax)

    savefig(fig, figures_dir / "figure_sp_segments.png")


def plot_interpretation_segments(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot full and low-frequency SP signals for interpretation segments.

    Args:
        df: DataFrame containing the processed electric potential table.
        figures_dir: Output directory for figure files.

    Notes:
        This function expects segment groups 1-2 and 3-4. If one group is not
        present, the corresponding subplot is annotated instead of failing.

        Y-limits honour the `interpretation_upstream` and
        `interpretation_downstream` exclusion entries. The x-axis is sample
        index, so x exclusions are given in samples.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    groups: list[tuple[Axes, pd.DataFrame, str, str, str]] = [
        (
            axes[0],
            df[df["segment_id"].isin([1, 2])].copy(),
            "interpretation_upstream",
            "Interpretation Segment 1-2",
            "No Segment 1-2 data found",
        ),
        (
            axes[1],
            df[df["segment_id"].isin([3, 4])].copy(),
            "interpretation_downstream",
            "Interpretation Segment 3-4",
            "No Segment 3-4 data found",
        ),
    ]

    for ax, d, key, title, empty_msg in groups:
        scaler = AxisScaler(key)

        if not d.empty:
            idx = np.arange(len(d))
            ax.plot(idx, d["SPmV_drift_corrected"], "k", lw=1.0, label="Full Signal")
            ax.plot(idx, d["DVL_lowfreq"], "r", lw=1.0, label="Low Frequency")
            scaler.add(idx, d["SPmV_drift_corrected"])
            scaler.add(idx, d["DVL_lowfreq"])
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                empty_msg,
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        ax.set_title(title)
        ax.set_ylabel("Voltage (mV)")
        ax.minorticks_on()
        ax.grid(alpha=0.2)
        scaler.apply(ax)

    axes[1].set_xlabel("Sample Index")

    savefig(fig, figures_dir / "figure_interpretation_segments.png")


def plot_integrated_potential(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot integrated electric potential components for both segment groups.

    Args:
        df: DataFrame containing the processed electric potential table.
        figures_dir: Output directory for figure files.

    Notes:
        This function expects segment groups 1-2 and 3-4. If one or both
        groups are missing, the available data are plotted and empty panels are
        annotated.

        Each panel scales independently using its own `integrated_<column>`
        exclusion entry. The x-axis is sample index.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)

    upstream = df[df["segment_id"].isin([1, 2])].copy()
    downstream = df[df["segment_id"].isin([3, 4])].copy()

    series: list[tuple[str, str]] = [
        ("V_full", "Integrated Electric Potential: Full Signal"),
        ("VL_lowfreq", "Integrated Electric Potential: Low Frequency"),
        ("VH_highfreq", "Integrated Electric Potential: High Frequency"),
        ("VN_noise", "Integrated Electric Potential: Noise"),
    ]

    for ax, (col, title) in zip(axes.flat, series):
        scaler = AxisScaler(f"integrated_{col}")

        if not upstream.empty:
            idx = np.arange(len(upstream))
            ax.plot(idx, upstream[col], "k", lw=1.0, label="Segment 1-2")
            scaler.add(idx, upstream[col])
        if not downstream.empty:
            idx = np.arange(len(downstream))
            ax.plot(idx, downstream[col], color="0.35", lw=1.0, label="Segment 3-4")
            scaler.add(idx, downstream[col])

        if upstream.empty and downstream.empty:
            ax.text(
                0.5,
                0.5,
                "No data found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        else:
            ax.legend()

        ax.set_title(title)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Voltage (mV)")
        ax.minorticks_on()
        ax.grid(alpha=0.2)
        scaler.apply(ax)

    savefig(fig, figures_dir / "figure_integrated_potential.png")


def plot_temp_cond(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot raw temperature and conductivity change by segment.

    Args:
        df: DataFrame containing the processed temperature/conductivity table.
        figures_dir: Output directory for figure files.

    Notes:
        Y-limits honour the `temperature` and `conductivity` exclusion entries.
        X exclusions are given in kilometres of segment distance.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    colors: dict[int, str] = {1: "k", 2: "r", 3: "g", 4: "b"}

    temp_scaler = AxisScaler("temperature")
    cond_scaler = AxisScaler("conductivity")

    for seg in sorted(df["segment_id"].unique()):
        d = df[df["segment_id"] == seg].copy()
        dist_km = d["segment_distance_m"] / 1000.0
        d_temp = d["temp_degC"] - d["temp_degC"].iloc[0]
        d_cond = d["cond_uS_cm"] - d["cond_uS_cm"].iloc[0]

        axes[0].plot(
            dist_km,
            d_temp,
            color=colors.get(int(seg), None),
            lw=1.2,
            label=f"Segment {seg}",
        )
        temp_scaler.add(dist_km, d_temp)

        axes[1].plot(
            dist_km,
            d_cond,
            color=colors.get(int(seg), None),
            lw=1.2,
            label=f"Segment {seg}",
        )
        cond_scaler.add(dist_km, d_cond)

    axes[0].set_title("Raw Temperature Change Relative to Segment Start")
    axes[0].set_xlabel("Segment Distance (km)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].minorticks_on()
    axes[0].grid(alpha=0.2)
    temp_scaler.apply(axes[0])

    axes[1].set_title("Raw Conductivity Change Relative to Segment Start")
    axes[1].set_xlabel("Segment Distance (km)")
    axes[1].set_ylabel("Conductivity (µS/cm)")
    axes[1].legend()
    axes[1].minorticks_on()
    axes[1].grid(alpha=0.2)
    cond_scaler.apply(axes[1])

    savefig(fig, figures_dir / "figure_temp_cond.png")


# =============================================================================
# DRIVER
# =============================================================================

def run_all_plots(config_path: str | Path = "config.toml") -> None:
    """Generate all available standard figures from processed output tables.

    Args:
        config_path: Path to the TOML configuration file.

    Notes:
        Missing CSV files do not stop execution. The script prints warnings to
        the terminal and skips any plots that depend on unavailable inputs.
    """
    cfg = load_config(config_path)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    ensure_dir(figures_dir)

    grad = read_csv_if_exists(processed_dir, "Gradient_Self_Potential_python.csv")
    pot = read_csv_if_exists(processed_dir, "Electric_Potential_python.csv")
    tc = read_csv_if_exists(processed_dir, "Temperature_Conductivity_python.csv")

    wrote_any = False

    if grad is not None:
        plot_gradient_sp_segments(grad, figures_dir)
        wrote_any = True
    else:
        print("[WARN] Skipped gradient SP plot.")

    if pot is not None:
        plot_interpretation_segments(pot, figures_dir)
        plot_integrated_potential(pot, figures_dir)
        wrote_any = True
    else:
        print("[WARN] Skipped interpretation and integrated potential plots.")

    if tc is not None:
        plot_temp_cond(tc, figures_dir)
        wrote_any = True
    else:
        print("[WARN] Skipped temperature/conductivity plots.")

    if wrote_any:
        print(f"[INFO] Wrote available figures to {figures_dir}")
    else:
        print(
            f"[WARN] No input CSV files found in {processed_dir}; "
            "no figures were created."
        )


def main() -> None:
    """Parse command-line arguments and generate figures."""
    parser = argparse.ArgumentParser(
        description="Plot processed self-potential outputs from config.toml"
    )
    parser.add_argument(
        "--config",
        default="config.toml",
        help="Path to TOML config file",
    )
    args = parser.parse_args()

    run_all_plots(args.config)


if __name__ == "__main__":
    main()
