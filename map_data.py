"""Interactive sample-location maps for WaSP self-potential surveys.

Reads the processed products written by ``data_processing.py`` and draws
matplotlib maps of every sample location on an Esri World Imagery basemap
(via contextily), coloured by measured value, annotated with segment distance
in kilometres, and finished with a scale bar and north arrow.

Maps produced (each written to ``figures_dir`` and shown interactively):

* Raw self-potential (mV)
* Drift-corrected self-potential (mV)
* Specific conductance (uS/cm)          - if temp/cond data present
* Water temperature (degC)              - if temp/cond data present

Interactivity
-------------
Hovering over any sample shows a tooltip with segment ID, distance along the
segment, survey distance, latitude/longitude and the mapped value. Clicking a
sample prints the same record to the console so it can be copied. This
requires a GUI backend (TkAgg or QtAgg); under a headless/Agg backend the
script still writes PNGs and simply skips the interactive display.

Configuration
-------------
Uses the same ``config.toml`` as the processing module:

[paths]
input_dir = '...'
processed_dir = '...'
figures_dir = '...'

[files]
sp_data = 'formatted_self_potential.csv'
drift_data = 'drift_self_potential.csv'
temp_cond_data = 'formatted_conductivity.csv'
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any


def _repair_proj_environment() -> None:
    """Point PROJ at this conda environment's database.

    On Windows, unrelated installs (PostgreSQL/PostGIS, ArcGIS, OSGeo4W) often
    leave a stale PROJ_LIB on the system, and rasterio then loads a proj.db
    that is too old to resolve EPSG codes. Must run before contextily/rasterio
    are imported.
    """
    candidates = [
        Path(sys.prefix) / "Library" / "share" / "proj",   # conda on Windows
        Path(sys.prefix) / "share" / "proj",               # conda on Linux/macOS
    ]
    for candidate in candidates:
        if (candidate / "proj.db").exists():
            existing = os.environ.get("PROJ_LIB", "")
            if existing and Path(existing).resolve() != candidate.resolve():
                print(f"NOTE: overriding stale PROJ_LIB ({existing}) -> {candidate}")
            os.environ["PROJ_LIB"] = str(candidate)
            os.environ["PROJ_DATA"] = str(candidate)
            return

    print(
        "WARNING: no proj.db found inside this environment; leaving PROJ_LIB "
        "as-is. Basemap tiles should still load since no reprojection is used."
    )


_repair_proj_environment()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tomllib
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

try:
    import contextily as ctx

    HAS_CONTEXTILY = True
except ImportError:  # pragma: no cover - depends on local environment
    ctx = None
    HAS_CONTEXTILY = False


# =============================================================================
# USER SETTINGS
# =============================================================================

# Colour map for the value gradient. 'RdYlBu_r' reads well over dark imagery;
# 'viridis' or 'coolwarm' are reasonable alternatives.
COLORMAP = "RdYlBu_r"

# Clip the colour scale to these percentiles so a few outliers do not flatten
# the gradient. Set to (0, 100) to use the full range.
COLOR_PERCENTILES = (2.0, 98.0)

# Approximate spacing between distance labels along each segment, in km.
LABEL_INTERVAL_KM = 0.25

# Marker size for sample points.
MARKER_SIZE = 26

# Fractional padding added around the data extent before fetching imagery.
EXTENT_PAD_FRAC = 0.12

# Basemap resolution. Higher = sharper but slower and more tiles fetched.
BASEMAP_ZOOM: int | str = "auto"

FIGURE_DPI = 200


# =============================================================================
# CONFIG
# =============================================================================

ConfigDict = dict[str, Any]


def load_config(config_path: str | Path = "config.toml") -> ConfigDict:
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        return tomllib.load(f)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

WEB_MERCATOR_R = 6_378_137.0


def lonlat_to_webmercator(
    lon_deg: np.ndarray, lat_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS84 lon/lat (degrees) to Web Mercator EPSG:3857 metres.

    Done inline so the script does not require pyproj. Latitudes are clamped
    to the Mercator valid range.
    """
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.clip(np.asarray(lat_deg, dtype=float), -85.05112878, 85.05112878)

    x = WEB_MERCATOR_R * np.radians(lon)
    y = WEB_MERCATOR_R * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def mercator_scale_factor(lat_deg: float) -> float:
    """Web Mercator inflates distances by 1/cos(lat).

    Multiply a true ground distance by this to get the equivalent length in
    projected map units.
    """
    return 1.0 / math.cos(math.radians(lat_deg))


def nice_bar_length(target_m: float) -> float:
    """Round a target scale-bar length down to a clean 1/2/5 x 10^n value."""
    if target_m <= 0:
        return 1.0
    exponent = math.floor(math.log10(target_m))
    base = 10.0**exponent
    for mult in (5.0, 2.0, 1.0):
        if mult * base <= target_m:
            return mult * base
    return base


# =============================================================================
# MAP DECORATIONS
# =============================================================================

def add_scale_bar(ax: Axes, center_lat_deg: float, n_blocks: int = 4) -> None:
    """Draw a checkered ground-distance scale bar in the lower-left.

    Corrects for Mercator distortion at the survey latitude, so the stated
    length is true ground distance rather than projected map units. The
    alternating black/white blocks stay legible over both dark water and
    bright bare ground in the imagery.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    map_span_m = x1 - x0
    scale = mercator_scale_factor(center_lat_deg)
    ground_span_m = map_span_m / scale

    bar_ground_m = nice_bar_length(0.25 * ground_span_m)
    bar_map_m = bar_ground_m * scale

    bx = x0 + 0.06 * (x1 - x0)
    by = y0 + 0.075 * (y1 - y0)
    bar_h = 0.011 * (y1 - y0)
    block_w = bar_map_m / n_blocks

    # Light backing panel so the bar and its labels read on any imagery
    ax.add_patch(
        Rectangle(
            (bx - 0.8 * block_w, by - 1.0 * bar_h),
            bar_map_m + 1.9 * block_w,
            4.6 * bar_h,
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            zorder=9,
        )
    )

    for i in range(n_blocks):
        ax.add_patch(
            Rectangle(
                (bx + i * block_w, by),
                block_w,
                bar_h,
                facecolor="black" if i % 2 == 0 else "white",
                edgecolor="black",
                linewidth=0.7,
                zorder=10,
            )
        )

    unit_div, unit = (1000.0, "km") if bar_ground_m >= 1000.0 else (1.0, "m")

    ax.text(
        bx,
        by + 1.4 * bar_h,
        "0",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=11,
    )
    ax.text(
        bx + bar_map_m,
        by + 1.4 * bar_h,
        f"{bar_ground_m / unit_div:g} {unit}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="black",
        zorder=11,
    )


def add_north_arrow(ax: Axes) -> None:
    """Draw a north arrow in the upper-right. Web Mercator north is up."""
    x_frac, y_base, y_tip = 0.945, 0.815, 0.885

    ax.add_patch(
        Rectangle(
            (x_frac - 0.030, y_base - 0.040),
            0.060,
            (y_tip - y_base) + 0.072,
            transform=ax.transAxes,
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
            zorder=9,
        )
    )

    # Stem
    ax.plot(
        [x_frac, x_frac],
        [y_base, y_tip - 0.014],
        transform=ax.transAxes,
        color="black",
        lw=1.8,
        solid_capstyle="butt",
        zorder=10,
    )

    # Solid triangular head, drawn explicitly so it never renders hollow
    head = Polygon(
        [
            (x_frac, y_tip + 0.014),
            (x_frac - 0.015, y_tip - 0.018),
            (x_frac + 0.015, y_tip - 0.018),
        ],
        closed=True,
        transform=ax.transAxes,
        facecolor="black",
        edgecolor="black",
        zorder=10,
    )
    ax.add_patch(head)

    ax.text(
        x_frac,
        y_base - 0.006,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="black",
        zorder=11,
    )


def add_distance_labels(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    segment_id: np.ndarray,
    distance_km: np.ndarray,
    interval_km: float,
) -> None:
    """Label points at roughly `interval_km` spacing along each segment."""
    for seg in np.unique(segment_id):
        mask = segment_id == seg
        d = distance_km[mask]
        xs = x[mask]
        ys = y[mask]

        if len(d) == 0:
            continue

        order = np.argsort(d)
        d, xs, ys = d[order], xs[order], ys[order]

        next_label = d[0]
        for i in range(len(d)):
            if d[i] + 1e-9 < next_label:
                continue
            ax.annotate(
                f"{d[i]:.2f} km",
                xy=(xs[i], ys[i]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=7,
                color="black",
                zorder=12,
                bbox=dict(
                    boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.75
                ),
            )
            next_label = d[i] + interval_km


def attach_hover(
    fig: Figure,
    ax: Axes,
    scatter: PathCollection,
    records: pd.DataFrame,
    value_label: str,
    value_fmt: str,
) -> None:
    """Wire up hover tooltips and click-to-print for the scatter points."""
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(16, 16),
        textcoords="offset points",
        fontsize=8,
        zorder=20,
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="0.3", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    annot.set_visible(False)

    def describe(idx: int) -> str:
        row = records.iloc[idx]
        lines = [
            f"Segment {int(row['segment_id'])}  ·  point {idx}",
            f"{value_label}: {row['value']:{value_fmt}}",
            f"Segment dist: {row['segment_distance_km']:.3f} km",
        ]
        if "survey_distance_km" in records.columns and pd.notna(
            row["survey_distance_km"]
        ):
            lines.append(f"Survey dist:  {row['survey_distance_km']:.3f} km")
        lines.append(f"Lat/Lon: {row['lat']:.6f}, {row['lon']:.6f}")
        return "\n".join(lines)

    def on_move(event) -> None:
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        contains, info = scatter.contains(event)
        if contains:
            idx = int(info["ind"][0])
            pos = scatter.get_offsets()[idx]
            annot.xy = (pos[0], pos[1])
            annot.set_text(describe(idx))
            annot.set_visible(True)
            fig.canvas.draw_idle()
        elif annot.get_visible():
            annot.set_visible(False)
            fig.canvas.draw_idle()

    def on_click(event) -> None:
        if event.inaxes is not ax:
            return
        contains, info = scatter.contains(event)
        if contains:
            print("\n" + describe(int(info["ind"][0])))

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event", on_click)


# =============================================================================
# MAP BUILDER
# =============================================================================

def make_map(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    figures_dir: Path,
    filename: str,
    value_fmt: str = ".2f",
    use_basemap: bool = True,
) -> Figure:
    """Draw one decorated sample map and save it to `figures_dir`."""
    frame = df.dropna(subset=["lon", "lat", value_col]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No plottable rows for '{value_col}'.")

    x, y = lonlat_to_webmercator(frame["lon"].to_numpy(), frame["lat"].to_numpy())
    values = frame[value_col].to_numpy(dtype=float)

    lo, hi = np.percentile(values, COLOR_PERCENTILES)
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        lo, hi = float(np.min(values)), float(np.max(values))
        if math.isclose(lo, hi):
            lo, hi = lo - 0.5, hi + 0.5

    fig, ax = plt.subplots(figsize=(10.5, 8.5))

    # Pad the extent so points are not jammed against the frame, and keep the
    # aspect square so the scale bar is valid in both directions.
    x_span = max(x.max() - x.min(), 1.0)
    y_span = max(y.max() - y.min(), 1.0)
    span = max(x_span, y_span) * (1.0 + 2.0 * EXTENT_PAD_FRAC)
    cx, cy = (x.max() + x.min()) / 2.0, (y.max() + y.min()) / 2.0

    ax.set_xlim(cx - span / 2.0, cx + span / 2.0)
    ax.set_ylim(cy - span / 2.0, cy + span / 2.0)
    ax.set_aspect("equal")

    if use_basemap and HAS_CONTEXTILY:
        try:
            # No `crs=` argument on purpose. Coordinates are already projected
            # to Web Mercator by lonlat_to_webmercator(), which is what
            # contextily assumes by default. Passing a CRS would trigger a
            # rasterio reprojection that needs proj.db and breaks on machines
            # with a conflicting PROJ install (PostGIS, ArcGIS, OSGeo4W).
            ctx.add_basemap(
                ax,
                source=ctx.providers.Esri.WorldImagery,
                zoom=BASEMAP_ZOOM,
                attribution_size=6,
            )
        except Exception as exc:  # network, tile server, zoom issues
            print(f"WARNING: basemap unavailable ({exc}); drawing without imagery.")
            ax.set_facecolor("0.15")
    else:
        if use_basemap and not HAS_CONTEXTILY:
            print("WARNING: contextily not installed; drawing without imagery.")
        ax.set_facecolor("0.15")

    # Faint track line so the survey path is readable between samples
    for seg in np.unique(frame["segment_id"].to_numpy()):
        mask = frame["segment_id"].to_numpy() == seg
        order = np.argsort(frame.loc[mask, "segment_distance_km"].to_numpy())
        ax.plot(
            x[mask][order],
            y[mask][order],
            color="white",
            lw=0.8,
            alpha=0.45,
            zorder=4,
        )

    scatter = ax.scatter(
        x,
        y,
        c=values,
        cmap=COLORMAP,
        vmin=lo,
        vmax=hi,
        s=MARKER_SIZE,
        edgecolors="black",
        linewidths=0.3,
        zorder=5,
    )

    add_distance_labels(
        ax,
        x,
        y,
        frame["segment_id"].to_numpy(),
        frame["segment_distance_km"].to_numpy(dtype=float),
        LABEL_INTERVAL_KM,
    )

    add_scale_bar(ax, center_lat_deg=float(frame["lat"].mean()))
    add_north_arrow(ax)

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    n_seg = frame["segment_id"].nunique()
    ax.text(
        0.015,
        0.015,
        f"{len(frame)} samples · {n_seg} segment(s)\n"
        f"colour clipped to {COLOR_PERCENTILES[0]:g}-{COLOR_PERCENTILES[1]:g} pct",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8),
        zorder=11,
    )

    records = frame.assign(value=values)
    attach_hover(fig, ax, scatter, records, colorbar_label, value_fmt)

    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / filename
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"Wrote {out_path}")

    return fig


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sp_frame(processed_dir: Path, input_dir: Path, sp_filename: str) -> pd.DataFrame:
    """Load SP points, preferring the processed gradient product.

    The processed file carries segment/survey distances and the drift-corrected
    series; the raw formatted file is a fallback with distances computed as NaN.
    """
    processed = processed_dir / "Gradient_Self_Potential_python.csv"

    if processed.exists():
        df = pd.read_csv(processed)
        df.columns = [str(c).strip() for c in df.columns]
        return pd.DataFrame(
            {
                "segment_id": pd.to_numeric(df["segment_id"], errors="coerce"),
                "lon": pd.to_numeric(df["x"], errors="coerce"),
                "lat": pd.to_numeric(df["y"], errors="coerce"),
                "segment_distance_km": pd.to_numeric(
                    df["segment_distance_km"], errors="coerce"
                ),
                "survey_distance_km": pd.to_numeric(
                    df["survey_distance_km"], errors="coerce"
                ),
                "raw_SP_mV": pd.to_numeric(df["raw_SP_mV"], errors="coerce"),
                "drift_corrected_SP_mV": pd.to_numeric(
                    df["drift_corrected_SP_mV"], errors="coerce"
                ),
            }
        )

    print(
        f"NOTE: {processed.name} not found; falling back to raw {sp_filename}. "
        "Run data_processing.py with write=True for drift-corrected maps."
    )
    raw = pd.read_csv(input_dir / sp_filename)
    raw.columns = [str(c).strip() for c in raw.columns]

    out = pd.DataFrame(
        {
            "segment_id": pd.to_numeric(raw["Segment_ID"], errors="coerce"),
            "lon": pd.to_numeric(raw["Longitude"], errors="coerce"),
            "lat": pd.to_numeric(raw["Latitude"], errors="coerce"),
            "raw_SP_mV": pd.to_numeric(
                raw["Measured_Voltage_millivolts"], errors="coerce"
            ),
        }
    )
    out["drift_corrected_SP_mV"] = np.nan
    out["segment_distance_km"] = compute_segment_distance_km(out)
    out["survey_distance_km"] = np.nan
    return out


def compute_segment_distance_km(df: pd.DataFrame) -> np.ndarray:
    """Haversine distance from each segment's first point, in km."""
    r_earth = 6_367_000.0
    dist = np.full(len(df), np.nan)

    for seg in df["segment_id"].dropna().unique():
        mask = (df["segment_id"] == seg).to_numpy()
        lat = df.loc[mask, "lat"].to_numpy(dtype=float)
        lon = df.loc[mask, "lon"].to_numpy(dtype=float)
        if len(lat) == 0:
            continue

        d2r = np.pi / 180.0
        lat_a, lon_a = d2r * lat[0], d2r * lon[0]
        lat_b, lon_b = d2r * lat, d2r * lon
        a = (
            np.sin((lat_b - lat_a) / 2.0) ** 2
            + np.cos(lat_a) * np.cos(lat_b) * np.sin((lon_b - lon_a) / 2.0) ** 2
        )
        dist[mask] = (r_earth * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))) / 1000.0

    return dist


def load_tc_frame(processed_dir: Path) -> pd.DataFrame | None:
    """Load the processed temperature/conductivity product, if it exists."""
    path = processed_dir / "Temperature_Conductivity_python.csv"
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    return pd.DataFrame(
        {
            "segment_id": pd.to_numeric(df["segment_id"], errors="coerce"),
            "lon": pd.to_numeric(df["x"], errors="coerce"),
            "lat": pd.to_numeric(df["y"], errors="coerce"),
            # Processed TC distances are in metres; convert for label consistency
            "segment_distance_km": pd.to_numeric(
                df["segment_distance_m"], errors="coerce"
            )
            / 1000.0,
            "survey_distance_km": pd.to_numeric(
                df["survey_distance_m"], errors="coerce"
            )
            / 1000.0,
            "temp_degC": pd.to_numeric(df["temp_degC"], errors="coerce"),
            "spec_cond_uS_cm": pd.to_numeric(df["spec_cond_uS_cm"], errors="coerce"),
        }
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(config_path: str | Path = "config.toml", show: bool = True,
         use_basemap: bool = True) -> None:
    cfg = load_config(config_path)

    input_dir = Path(cfg["paths"]["input_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])

    sp = load_sp_frame(processed_dir, input_dir, cfg["files"]["sp_data"])
    print(f"Loaded {len(sp)} SP samples across {sp['segment_id'].nunique()} segment(s)")

    make_map(
        sp,
        value_col="raw_SP_mV",
        title="Self-Potential Sample Locations - Raw",
        colorbar_label="Measured voltage (mV)",
        figures_dir=figures_dir,
        filename="map_sp_raw.png",
        use_basemap=use_basemap,
    )

    if sp["drift_corrected_SP_mV"].notna().any():
        make_map(
            sp,
            value_col="drift_corrected_SP_mV",
            title="Self-Potential Sample Locations - Drift Corrected",
            colorbar_label="Drift-corrected voltage (mV)",
            figures_dir=figures_dir,
            filename="map_sp_drift_corrected.png",
            use_basemap=use_basemap,
        )

    tc = load_tc_frame(processed_dir)
    if tc is not None and not tc.empty:
        print(f"Loaded {len(tc)} temperature/conductivity samples")

        make_map(
            tc,
            value_col="spec_cond_uS_cm",
            title="Specific Conductance Sample Locations",
            colorbar_label="Specific conductance (uS/cm)",
            figures_dir=figures_dir,
            filename="map_specific_conductance.png",
            value_fmt=".1f",
            use_basemap=use_basemap,
        )

        make_map(
            tc,
            value_col="temp_degC",
            title="Surface Water Temperature Sample Locations",
            colorbar_label="Temperature (degC)",
            figures_dir=figures_dir,
            filename="map_temperature.png",
            use_basemap=use_basemap,
        )
    else:
        print("No processed temperature/conductivity product found; skipping.")

    if show:
        backend = plt.get_backend().lower()
        if backend == "agg":
            print(
                "Backend is 'agg' (non-interactive). PNGs were written, but "
                "hover tooltips need a GUI backend such as TkAgg or QtAgg."
            )
        else:
            print("\nHover a sample for details; click to print it to the console.")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.toml", help="Path to config.toml")
    parser.add_argument("--no-show", action="store_true", help="Save figures only")
    parser.add_argument(
        "--no-basemap", action="store_true", help="Skip Esri imagery (offline use)"
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        show=not args.no_show,
        use_basemap=not args.no_basemap,
    )
