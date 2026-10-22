"""
preproc_no_gnss.py

WaSP preprocessing for surveys WITHOUT a GNSS (Arrow) log.

Produces the same outputs as ``preproc.py``:
    formatted_SP_data.csv       (drift readings, with Longitude/Latitude)
    formatted_VDrift.csv        (eddy readings, only if a mode_flag column exists)
    formatted_conductivity.csv  (optional - skipped when no conductivity data)

Positions come from a polyline shapefile instead of GNSS:
    * The electrode CSV has a ``segment`` column (e.g. 1-4).
    * Each segment number matches the ID field (default ``id``) in the shapefile.
    * The drift readings for a segment are spaced equidistantly along that
      segment's polyline (first reading at the line start, last at the line end),
      and the coordinates of each placed point are written out.

Conductivity (if included) is located by time: each conductivity record is
matched to the nearest placed electrode point within +/- CONDUCTIVITY_MAX_TIME_DIFF_SECONDS.

Electrode CSV columns
---------------------
Required: Time, Reading, segment
Optional: mode_flag  ("drift" / "eddy"). If absent, every row is treated as drift.

Dependencies: pandas, numpy, geopandas, shapely
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union


# =============================================================================
# USER SETTINGS - EDIT THESE
# =============================================================================
RAW_DIRECTORY = Path(r'C:\Users\CND367\Documents\WaSP\jeff_20261020\raw')
ELECTRODE_FILE = RAW_DIRECTORY / 'floats.csv'
SEGMENT_SHAPEFILE = RAW_DIRECTORY / 'shp' / 'jeff_FLOAT.shp'
CONDUCTIVITY_FILE = None

PROC_DIRECTORY = Path(r'C:\Users\CND367\Documents\WaSP\jeff_20261020')
SP_OUTPUT_FILE = PROC_DIRECTORY / 'formatted_SP_data.csv'
VDRIFT_OUTPUT_FILE = PROC_DIRECTORY / 'formatted_VDrift.csv'
CONDUCTIVITY_OUTPUT_FILE = PROC_DIRECTORY / 'formatted_conductivity.csv'

# Optional QA layer of the placed SP points (open in ArcGIS Pro). Set to None to skip.
QA_POINTS_FILE: Optional[Path] = PROC_DIRECTORY / 'placed_SP_points.gpkg'
QA_POINTS_FILE: None

# --- Conductivity ------------------------------------------------------------
# False -> skip conductivity entirely (no file needed, no output written).
# True  -> read CONDUCTIVITY_FILE; if the file is missing, a warning is printed
#          and conductivity is skipped.
INCLUDE_CONDUCTIVITY = False
CONDUCTIVITY_MAX_TIME_DIFF_SECONDS = 5.0

# --- Electrode CSV columns ---------------------------------------------------
ELECTRODE_TIME_COLUMN = "Time"
ELECTRODE_READING_COLUMN = "Reading"
ELECTRODE_SEGMENT_COLUMN = "Segment"
ELECTRODE_MODE_COLUMN = "mode_flag"     # optional; if missing, all rows = drift
ELECTRODE_TIME_FORMAT = "%m/%d/%Y %H:%M:%S.%f"   # falls back to auto-parse
VOLTAGE_TIME_OFFSET_SECONDS = 0.0

# --- Shapefile ---------------------------------------------------------------
SHAPEFILE_ID_FIELD = "id"               # case-insensitive match

# Segments whose polyline was digitized opposite to the direction of travel.
# Readings on these segments are placed from the line END toward the START.
REVERSED_SEGMENTS: set[int] = set()

# Order readings within a segment by time ("time") or by CSV row order ("file").
WITHIN_SEGMENT_ORDER = "time"

# Measurement_Number: restart at 1 for each segment (True) or run continuously (False)
NUMBER_PER_SEGMENT = True

# Optional start/end location names per segment. Segments not listed use the defaults.
DEFAULT_START_LOCATION = "Unknown Start"
DEFAULT_END_LOCATION = "Unknown End"
SEGMENT_LOCATIONS: dict[int, tuple[str, str]] = {
    # 1: ("Bridge", "Diversion"),
    # 2: ("Diversion", "Gage"),
}


# =============================================================================
# READERS
# =============================================================================
def find_column(df: pd.DataFrame, name: str, required: bool = True) -> Optional[str]:
    """Return the actual column name matching ``name`` (case/whitespace-insensitive)."""
    target = name.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == target:
            return col
    if required:
        raise ValueError(f"Column '{name}' not found. Columns found: {list(df.columns)}")
    return None


def read_electrode_data(path: Path) -> pd.DataFrame:
    """
    Read the electrode CSV.

    Returns columns: timestamp, Measured_Voltage_millivolts, segment, mode_flag, file_order
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    time_col = find_column(df, ELECTRODE_TIME_COLUMN)
    reading_col = find_column(df, ELECTRODE_READING_COLUMN)
    segment_col = find_column(df, ELECTRODE_SEGMENT_COLUMN)
    mode_col = find_column(df, ELECTRODE_MODE_COLUMN, required=False)

    out = pd.DataFrame({"file_order": np.arange(len(df))})

    time_str = df[time_col].astype(str).str.strip()
    ts = pd.to_datetime(time_str, format=ELECTRODE_TIME_FORMAT, errors="coerce")
    if ts.isna().all():
        ts = pd.to_datetime(time_str, errors="coerce")
    out["timestamp"] = ts + pd.to_timedelta(VOLTAGE_TIME_OFFSET_SECONDS, unit="s")

    out["Measured_Voltage_millivolts"] = pd.to_numeric(df[reading_col], errors="coerce")
    out["segment"] = pd.to_numeric(df[segment_col], errors="coerce")

    if mode_col is not None:
        out["mode_flag"] = df[mode_col].astype(str).str.strip().str.lower()
    else:
        out["mode_flag"] = "drift"

    n_before = len(out)
    out = out.dropna(subset=["timestamp", "Measured_Voltage_millivolts", "segment"]).copy()
    dropped = n_before - len(out)
    if dropped:
        print(f"WARNING: dropped {dropped} electrode rows with bad time/reading/segment.")
    if out.empty:
        raise ValueError("No valid electrode rows found.")

    out["segment"] = out["segment"].astype(int)
    out["timestamp"] = pd.to_datetime(out["timestamp"]).astype("datetime64[ns]")
    return out.reset_index(drop=True)


def read_segment_lines(path: Path) -> tuple[dict[int, LineString], gpd.GeoDataFrame]:
    """
    Read the polyline shapefile and return ({segment_id: LineString in metric CRS}, gdf_metric).

    Features sharing the same ID are merged. Lines are measured in a projected CRS
    (the shapefile's own CRS if projected, otherwise an estimated UTM zone).
    """
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(
            "Shapefile has no CRS (.prj missing). Define the projection in ArcGIS first."
        )

    id_col = find_column(gdf, SHAPEFILE_ID_FIELD)
    gdf = gdf[[id_col, "geometry"]].rename(columns={id_col: "seg_id"})
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    gdf["seg_id"] = pd.to_numeric(gdf["seg_id"], errors="coerce")
    gdf = gdf.dropna(subset=["seg_id"])
    gdf["seg_id"] = gdf["seg_id"].astype(int)

    if gdf.crs.is_geographic:
        gdf = gdf.to_crs(gdf.estimate_utm_crs())

    lines: dict[int, LineString] = {}
    for seg_id, grp in gdf.groupby("seg_id"):
        geom = unary_union(list(grp.geometry))
        if isinstance(geom, MultiLineString):
            geom = linemerge(geom)
        if isinstance(geom, MultiLineString):
            # Disconnected parts: keep the longest and warn.
            parts = sorted(geom.geoms, key=lambda g: g.length, reverse=True)
            print(
                f"WARNING: segment {seg_id} has {len(parts)} disconnected parts; "
                f"using the longest ({parts[0].length:.1f} m). Fix gaps in ArcGIS for best results."
            )
            geom = parts[0]
        if not isinstance(geom, LineString):
            raise ValueError(f"Segment {seg_id} geometry is not a polyline ({geom.geom_type}).")
        lines[int(seg_id)] = geom

    return lines, gdf


def read_conductivity_data(path: Path) -> pd.DataFrame:
    """Same parser as preproc.py."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(*prefixes: str) -> str:
        for prefix in prefixes:
            p = prefix.strip().lower()
            for col in df.columns:
                if str(col).strip().lower().startswith(p):
                    return col
        raise ValueError(
            f"Conductivity file has no column starting with {prefixes}. "
            f"Columns found: {list(df.columns)}"
        )

    datetime_col = find_col("Date-Time", "Date- Time", "Date Time")
    temperature_col = find_col("Temperature")
    spcond_col = find_col("Specific Conductivity", "Specific Conductance")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[datetime_col].astype(str).str.strip(), errors="coerce")
    df["MeasuredSurfaceWaterTemperatureCelsius"] = pd.to_numeric(df[temperature_col], errors="coerce")
    df["MeasuredSurfaceWaterConductivitymicrosiemenspercm"] = pd.to_numeric(df[spcond_col], errors="coerce")

    df = df.dropna(
        subset=[
            "timestamp",
            "MeasuredSurfaceWaterTemperatureCelsius",
            "MeasuredSurfaceWaterConductivitymicrosiemenspercm",
        ]
    ).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    return df.sort_values("timestamp").reset_index(drop=True)[
        [
            "timestamp",
            "MeasuredSurfaceWaterTemperatureCelsius",
            "MeasuredSurfaceWaterConductivitymicrosiemenspercm",
        ]
    ]


# =============================================================================
# HELPERS
# =============================================================================
def format_measurement_time(ts: pd.Timestamp) -> str:
    return f"{ts.hour}:{ts.minute:02d}:{ts.second:02d}"


def segment_locations(seg_id: int) -> tuple[str, str]:
    return SEGMENT_LOCATIONS.get(seg_id, (DEFAULT_START_LOCATION, DEFAULT_END_LOCATION))


def assign_eddy_sequence_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Label contiguous eddy runs (in file order) as eddy_1, eddy_2, ..."""
    df = df.copy()
    gap = df["file_order"].diff().fillna(0)  # first row starts sequence 1
    df["eddy_sequence_id"] = (gap != 1).cumsum()
    df["DRIFT_MEASUREMENT_LOCATION"] = "eddy_" + df["eddy_sequence_id"].astype(int).astype(str)
    return df


def place_points_on_line(line: LineString, n: int, reverse: bool = False) -> list[Point]:
    """
    Return n points spaced equidistantly along ``line``.
    First point = line start, last point = line end (or reversed).
    A single reading is placed at the line midpoint.
    """
    if n <= 0:
        return []
    length = line.length
    distances = np.array([length / 2.0]) if n == 1 else np.linspace(0.0, length, n)
    if reverse:
        distances = length - distances
    return [line.interpolate(float(d)) for d in distances]


def place_drift_readings(
    drift_df: pd.DataFrame, lines: dict[int, LineString], line_crs
) -> gpd.GeoDataFrame:
    """Place drift readings equidistantly on their segment polylines; return lon/lat."""
    order_col = "timestamp" if WITHIN_SEGMENT_ORDER.lower() == "time" else "file_order"
    placed = []

    for seg_id, grp in drift_df.groupby("segment", sort=True):
        if seg_id not in lines:
            print(
                f"WARNING: segment {seg_id} ({len(grp)} readings) has no matching "
                f"'{SHAPEFILE_ID_FIELD}' in the shapefile - skipped."
            )
            continue
        grp = grp.sort_values([order_col, "file_order"]).copy()
        line = lines[seg_id]
        pts = place_points_on_line(line, len(grp), reverse=seg_id in REVERSED_SEGMENTS)
        grp["geometry"] = pts
        grp["Distance_Along_Segment_m"] = [line.project(p) for p in pts]
        spacing = line.length / (len(grp) - 1) if len(grp) > 1 else 0.0
        print(
            f"Segment {seg_id}: {len(grp)} readings over {line.length:.1f} m "
            f"(spacing {spacing:.2f} m){' [reversed]' if seg_id in REVERSED_SEGMENTS else ''}"
        )
        placed.append(grp)

    if not placed:
        raise ValueError("No drift readings could be placed on any segment.")

    gdf = gpd.GeoDataFrame(pd.concat(placed), geometry="geometry", crs=line_crs)
    gdf_wgs = gdf.to_crs(epsg=4326)
    gdf_wgs["Longitude"] = gdf_wgs.geometry.x
    gdf_wgs["Latitude"] = gdf_wgs.geometry.y
    return gdf_wgs


def add_measurement_numbers(df: pd.DataFrame, group_col: str, order_cols: list[str]) -> pd.Series:
    df = df.sort_values(order_cols)
    if NUMBER_PER_SEGMENT:
        return df.groupby(group_col).cumcount() + 1
    return pd.Series(range(1, len(df) + 1), index=df.index)


def check_segment_times(df: pd.DataFrame) -> None:
    """Warn if segment time windows overlap (usually a wrong segment number)."""
    windows = df.groupby("segment")["timestamp"].agg(["min", "max"]).sort_values("min")
    print("\n--- Segment time windows (local) ---")
    for seg_id, row in windows.iterrows():
        span = (row["max"] - row["min"]).total_seconds() / 60.0
        print(f"Segment {seg_id}: {row['min']} -> {row['max']} ({span:.1f} min)")
    prev_end, prev_id = None, None
    for seg_id, row in windows.iterrows():
        if prev_end is not None and row["min"] < prev_end:
            print(f"WARNING: segment {seg_id} starts before segment {prev_id} ends.")
        prev_end, prev_id = row["max"], seg_id
    print("------------------------------------\n")


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    PROC_DIRECTORY.mkdir(parents=True, exist_ok=True)

    electrode_df = read_electrode_data(ELECTRODE_FILE)
    lines, lines_gdf = read_segment_lines(SEGMENT_SHAPEFILE)
    print(f"Shapefile segments found: {sorted(lines)}  (measured in {lines_gdf.crs.name})")
    check_segment_times(electrode_df)

    print("mode_flag counts:")
    print(electrode_df["mode_flag"].value_counts().to_string())

    # -------------------------------------------------------------------------
    # SP output: drift readings placed equidistantly on each segment polyline
    # -------------------------------------------------------------------------
    drift_df = electrode_df[electrode_df["mode_flag"] == "drift"].copy()
    placed = place_drift_readings(drift_df, lines, lines_gdf.crs)

    order_col = "timestamp" if WITHIN_SEGMENT_ORDER.lower() == "time" else "file_order"
    placed = placed.sort_values(["segment", order_col, "file_order"])
    placed["Measurement_Number"] = add_measurement_numbers(
        placed, "segment", ["segment", order_col, "file_order"]
    )
    placed["Segment_ID"] = placed["segment"]
    placed["Segment_Start_Location"] = placed["segment"].map(lambda s: segment_locations(s)[0])
    placed["Segment_End_Location"] = placed["segment"].map(lambda s: segment_locations(s)[1])
    placed["Measurement_Time"] = placed["timestamp"].apply(format_measurement_time)

    sp_output_df = pd.DataFrame(placed[
        [
            "Segment_ID",
            "Segment_Start_Location",
            "Segment_End_Location",
            "Measurement_Number",
            "Measurement_Time",
            "Longitude",
            "Latitude",
            "Measured_Voltage_millivolts",
        ]
    ])
    sp_output_df.to_csv(SP_OUTPUT_FILE, index=False)

    if QA_POINTS_FILE is not None:
        qa = placed[
            ["Segment_ID", "Measurement_Number", "Measurement_Time",
             "Measured_Voltage_millivolts", "Distance_Along_Segment_m", "geometry"]
        ].copy()
        qa.to_file(QA_POINTS_FILE, driver="GPKG")

    # -------------------------------------------------------------------------
    # VDRIFT output: eddy readings (only if the CSV has a mode_flag column)
    # -------------------------------------------------------------------------
    eddy_df = electrode_df[electrode_df["mode_flag"] == "eddy"].sort_values("file_order").copy()
    if not eddy_df.empty:
        eddy_df = assign_eddy_sequence_labels(eddy_df)
        eddy_df["MeasurementNumber"] = eddy_df.groupby("eddy_sequence_id").cumcount() + 1
        eddy_df["MeasurementTime"] = eddy_df["timestamp"].apply(format_measurement_time)
        vdrift_output_df = eddy_df.rename(
            columns={
                "segment": "SegmentID",
                "DRIFT_MEASUREMENT_LOCATION": "DriftMeasurementLocation",
                "Measured_Voltage_millivolts": "MeasuredVoltagemillivolts",
            }
        )[
            ["SegmentID", "MeasurementNumber", "DriftMeasurementLocation",
             "MeasurementTime", "MeasuredVoltagemillivolts"]
        ]
    else:
        vdrift_output_df = pd.DataFrame(
            columns=["SegmentID", "MeasurementNumber", "DriftMeasurementLocation",
                     "MeasurementTime", "MeasuredVoltagemillivolts"]
        )
    vdrift_output_df.to_csv(VDRIFT_OUTPUT_FILE, index=False)

    # -------------------------------------------------------------------------
    # Conductivity output (optional): nearest placed SP point in time
    # -------------------------------------------------------------------------
    conductivity_rows = 0
    conductivity_output_df: Optional[pd.DataFrame] = None
    if not INCLUDE_CONDUCTIVITY:
        print("\nConductivity: excluded (INCLUDE_CONDUCTIVITY = False).")
    elif CONDUCTIVITY_FILE is None or not Path(CONDUCTIVITY_FILE).exists():
        print(f"\nWARNING: conductivity file not found ({CONDUCTIVITY_FILE}); skipping conductivity.")
    else:
        conductivity_df = read_conductivity_data(CONDUCTIVITY_FILE)
        conductivity_rows = len(conductivity_df)

        track = (
            pd.DataFrame(placed[["timestamp", "Longitude", "Latitude", "segment"]])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        cond = pd.merge_asof(
            conductivity_df,
            track,
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(seconds=CONDUCTIVITY_MAX_TIME_DIFF_SECONDS),
        ).dropna(subset=["Longitude", "Latitude"]).copy()
        cond["segment"] = cond["segment"].astype(int)

        cond["MeasurementNumber"] = add_measurement_numbers(cond, "segment", ["segment", "timestamp"])
        cond = cond.sort_values(["segment", "timestamp"])
        cond["SegmentID"] = cond["segment"]
        cond["SegmentStartLocation"] = cond["segment"].map(lambda s: segment_locations(s)[0])
        cond["SegmentEndLocation"] = cond["segment"].map(lambda s: segment_locations(s)[1])
        cond["MeasurementTime"] = cond["timestamp"].apply(format_measurement_time)

        conductivity_output_df = cond[
            [
                "SegmentID",
                "SegmentStartLocation",
                "SegmentEndLocation",
                "MeasurementNumber",
                "MeasurementTime",
                "Longitude",
                "Latitude",
                "MeasuredSurfaceWaterTemperatureCelsius",
                "MeasuredSurfaceWaterConductivitymicrosiemenspercm",
            ]
        ]
        conductivity_output_df.to_csv(CONDUCTIVITY_OUTPUT_FILE, index=False)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\nDone.")
    print(f"Electrode rows: {len(electrode_df)}")
    print(f"SP rows written (drift, placed on polylines): {len(sp_output_df)}")
    print(f"VDRIFT rows written (eddy): {len(vdrift_output_df)}")
    if conductivity_output_df is not None:
        print(f"Conductivity rows: {conductivity_rows}")
        print(
            f"Matched conductivity rows (+/- {CONDUCTIVITY_MAX_TIME_DIFF_SECONDS:g} s): "
            f"{len(conductivity_output_df)}"
        )
        print(f"Conductivity output: {CONDUCTIVITY_OUTPUT_FILE}")
    print(f"SP output: {SP_OUTPUT_FILE}")
    print(f"VDRIFT output: {VDRIFT_OUTPUT_FILE}")
    if QA_POINTS_FILE is not None:
        print(f"QA points: {QA_POINTS_FILE}")


if __name__ == "__main__":
    main()
