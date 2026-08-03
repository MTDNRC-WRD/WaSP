"""Generate figures from processed self-potential output tables.

This module reads processed CSV products defined by a TOML configuration file
and generates a standard set of figures for self-potential, integrated
electric potential, temperature, and conductivity. Missing CSV inputs are
treated as optional: the script prints a terminal warning and skips any plots
that depend on unavailable files.

The main entry point is `main()`, which parses a config path from the command
line and writes the figures into the configured figures directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import tomllib
from matplotlib.figure import Figure

ConfigDict = dict[str, Any]


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


def plot_gradient_sp_segments(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot drift-corrected gradient self-potential by segment.

    Args:
        df: DataFrame containing the processed gradient SP table.
        figures_dir: Output directory for figure files.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    colors: dict[int, str] = {1: "k", 2: "b", 3: "c", 4: "m"}
    labels: dict[int, str] = {
        1: "Segment 1",
        2: "Segment 2",
        3: "Segment 3",
        4: "Segment 4",
    }

    for seg in sorted(df["segment_id"].unique()):
        d = df[df["segment_id"] == seg].copy()
        ax.plot(
            d["segment_distance_km"],
            d["drift_corrected_SP_mV"],
            color=colors.get(int(seg), None),
            lw=1.2,
            label=labels.get(int(seg), f"Segment {seg}"),
        )

    ax.set_xlabel("Segment Distance (km)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Drift-Corrected Gradient SP by Segment")
    ax.legend()
    ax.minorticks_on()
    ax.grid(alpha=0.2)

    savefig(fig, figures_dir / "figure_sp_segments.png")


def plot_interpretation_segments(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot full and low-frequency SP signals for interpretation segments.

    Args:
        df: DataFrame containing the processed electric potential table.
        figures_dir: Output directory for figure files.

    Notes:
        This function expects segment groups 1-2 and 3-4. If one group is not
        present, the corresponding subplot is annotated instead of failing.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    upstream = df[df["segment_id"].isin([1, 2])].copy()
    downstream = df[df["segment_id"].isin([3, 4])].copy()

    if not upstream.empty:
        axes[0].plot(
            range(len(upstream)),
            upstream["SPmV_drift_corrected"],
            "k",
            lw=1.0,
            label="Full Signal",
        )
        axes[0].plot(
            range(len(upstream)),
            upstream["DVL_lowfreq"],
            "r",
            lw=1.0,
            label="Low Frequency",
        )
        axes[0].legend()
    else:
        axes[0].text(
            0.5,
            0.5,
            "No Segment 1-2 data found",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    axes[0].set_title("Interpretation Segment 1-2")
    axes[0].set_ylabel("Voltage (mV)")
    axes[0].minorticks_on()
    axes[0].grid(alpha=0.2)

    if not downstream.empty:
        axes[1].plot(
            range(len(downstream)),
            downstream["SPmV_drift_corrected"],
            "k",
            lw=1.0,
            label="Full Signal",
        )
        axes[1].plot(
            range(len(downstream)),
            downstream["DVL_lowfreq"],
            "r",
            lw=1.0,
            label="Low Frequency",
        )
        axes[1].legend()
    else:
        axes[1].text(
            0.5,
            0.5,
            "No Segment 3-4 data found",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )

    axes[1].set_title("Interpretation Segment 3-4")
    axes[1].set_xlabel("Sample Index")
    axes[1].set_ylabel("Voltage (mV)")
    axes[1].minorticks_on()
    axes[1].grid(alpha=0.2)

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
        if not upstream.empty:
            ax.plot(
                range(len(upstream)),
                upstream[col],
                "k",
                lw=1.0,
                label="Segment 1-2",
            )
        if not downstream.empty:
            ax.plot(
                range(len(downstream)),
                downstream[col],
                color="0.35",
                lw=1.0,
                label="Segment 3-4",
            )
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

    savefig(fig, figures_dir / "figure_integrated_potential.png")


def plot_temp_cond(df: pd.DataFrame, figures_dir: Path) -> None:
    """Plot raw temperature and conductivity change by segment.

    Args:
        df: DataFrame containing the processed temperature/conductivity table.
        figures_dir: Output directory for figure files.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    colors: dict[int, str] = {1: "k", 2: "r", 3: "g", 4: "b"}

    for seg in sorted(df["segment_id"].unique()):
        d = df[df["segment_id"] == seg].copy()

        axes[0].plot(
            d["segment_distance_m"] / 1000.0,
            d["temp_degC"] - d["temp_degC"].iloc[0],
            color=colors.get(int(seg), None),
            lw=1.2,
            label=f"Segment {seg}",
        )

        axes[1].plot(
            d["segment_distance_m"] / 1000.0,
            d["cond_uS_cm"] - d["cond_uS_cm"].iloc[0],
            color=colors.get(int(seg), None),
            lw=1.2,
            label=f"Segment {seg}",
        )

    axes[0].set_title("Raw Temperature Change Relative to Segment Start")
    axes[0].set_xlabel("Segment Distance (km)")
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].legend()
    axes[0].minorticks_on()
    axes[0].grid(alpha=0.2)

    axes[1].set_title("Raw Conductivity Change Relative to Segment Start")
    axes[1].set_xlabel("Segment Distance (km)")
    axes[1].set_ylabel("Conductivity (µS/cm)")
    axes[1].legend()
    axes[1].minorticks_on()
    axes[1].grid(alpha=0.2)

    savefig(fig, figures_dir / "figure_temp_cond.png")


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
        print(f"[WARN] No input CSV files found in {processed_dir}; no figures were created.")


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