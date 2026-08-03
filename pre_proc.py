import pandas as pd


# =============================================================================
# USER SETTINGS - EDIT THESE
# =============================================================================

ARROW_FILE = r"C:\Users\CND367\Documents\WaSP\test2\raw\arrow_log_20260731.csv"
VOLTAGE_FILE = r"C:\Users\CND367\Documents\WaSP\test2\raw\voltage_log.csv"
CONDUCTIVITY_FILE = r"C:\Users\CND367\Documents\WaSP\test2\raw\conductivity.csv"

SP_OUTPUT_FILE = r"C:\Users\CND367\Documents\WaSP\test2\formatted_self_potential.csv"
VDRIFT_OUTPUT_FILE = r"C:\Users\CND367\Documents\WaSP\test2\drift_self_potential.csv"
CONDUCTIVITY_OUTPUT_FILE = r"C:\Users\CND367\Documents\WaSP\test2\formatted_conductivity.csv"

# Choose which Arrow timestamp column to use:
#   "utc"   -> use pc_utc_time and convert to local time
#   "local" -> use pc_local_time directly
ARROW_TIME_MODE = "local"

TIMEZONE = "MDT"   # only used when ARROW_TIME_MODE = "utc"
UTC_TO_LOCAL_HOURS = -6 if TIMEZONE == "MDT" else -7

# The voltage file uses shorthand wall-clock times like 25:32.7, 55:04.4, 16:33.4.
# These are mm:ss.s and roll over every hour, so they are unwrapped in file order.
# Set this to the FULL LOCAL datetime of the FIRST voltage record AS IT APPEARS
# IN THE FILE (first row, not the earliest sorted value).
VOLTAGE_START_LOCAL_DATETIME = "2026-07-31 11:25:32"

# Optional extra manual time shifts
GNSS_TIME_OFFSET_SECONDS = 0.0
VOLTAGE_TIME_OFFSET_SECONDS = 0.0

# Maximum allowed time difference for matching a voltage point to nearest GNSS point
MAX_TIME_DIFF_SECONDS = 2.0

SEGMENT_ID = 1
SEGMENT_START_LOCATION = "Unknown Start"
SEGMENT_END_LOCATION = "Unknown End"


# =============================================================================
# FUNCTIONS
# =============================================================================

def read_arrow_data(path):
    df = pd.read_csv(path)

    required = ["lat_dd", "lon_dd", "mode_flag"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Arrow file is missing required columns: {missing}")

    df = df.copy()
    df["mode_flag"] = df["mode_flag"].astype(str).str.strip().str.lower()

    if ARROW_TIME_MODE.lower() == "local":
        if "pc_local_time" not in df.columns:
            raise ValueError(
                "ARROW_TIME_MODE is 'local' but pc_local_time was not found in the Arrow file."
            )

        df["timestamp"] = pd.to_datetime(
            df["pc_local_time"].astype(str).str.strip(),
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp", "lat_dd", "lon_dd", "mode_flag"]).copy()

        if isinstance(df["timestamp"].dtype, pd.DatetimeTZDtype):
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        df["timestamp"] = (
            pd.to_datetime(df["timestamp"])
            + pd.to_timedelta(GNSS_TIME_OFFSET_SECONDS, unit="s")
        )

    elif ARROW_TIME_MODE.lower() == "utc":
        if "pc_utc_time" not in df.columns:
            raise ValueError(
                "ARROW_TIME_MODE is 'utc' but pc_utc_time was not found in the Arrow file."
            )

        df["timestamp_utc"] = pd.to_datetime(
            df["pc_utc_time"].astype(str).str.strip(),
            utc=True,
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp_utc", "lat_dd", "lon_dd", "mode_flag"]).copy()

        df["timestamp"] = (
            df["timestamp_utc"]
            + pd.to_timedelta(UTC_TO_LOCAL_HOURS, unit="h")
            + pd.to_timedelta(GNSS_TIME_OFFSET_SECONDS, unit="s")
        )

        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    else:
        raise ValueError("ARROW_TIME_MODE must be either 'utc' or 'local'.")

    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "lon_dd", "lat_dd", "mode_flag"]].rename(
        columns={
            "lon_dd": "Longitude",
            "lat_dd": "Latitude",
        }
    )


def parse_elapsed_mmss(value):
    """
    Parse shorthand mm:ss.s strings like:
      25:32.7  -> 1532.7 seconds
      55:04.4  -> 3304.4 seconds
      16:33.4  ->  993.4 seconds
    Returns seconds-within-the-hour as float, or None if unparseable.
    """
    s = str(value).strip()

    if ":" not in s:
        return None

    parts = s.split(":")
    if len(parts) != 2:
        return None

    try:
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except ValueError:
        return None


def read_voltage_data(path):
    """
    Reads the voltage log. The 'Time' column holds wall-clock mm:ss.s values that
    wrap every hour, so rows are processed in FILE ORDER and each decrease in the
    mm:ss value is treated as an hour rollover.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = ["Time", "Reading"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Voltage file is missing required columns: {missing}")

    df = df.copy()

    # Do NOT sort here - file order carries the hour rollover information.
    df["mmss_seconds"] = df["Time"].apply(parse_elapsed_mmss)
    df["Reading"] = pd.to_numeric(df["Reading"], errors="coerce")

    df = df.dropna(subset=["mmss_seconds", "Reading"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid voltage time/reading rows found.")

    # Unwrap hour rollovers: each time the mm:ss value decreases, add 3600 s.
    elapsed = []
    wraps = 0
    prev = None
    for v in df["mmss_seconds"]:
        if prev is not None and v < prev - 1.0:   # 1 s guard against jitter
            wraps += 1
        elapsed.append(v + 3600.0 * wraps)
        prev = v

    df["elapsed_seconds"] = elapsed

    start_local = pd.to_datetime(VOLTAGE_START_LOCAL_DATETIME, errors="raise")
    first_elapsed = df["elapsed_seconds"].iloc[0]

    df["timestamp"] = (
        start_local
        + pd.to_timedelta(df["elapsed_seconds"] - first_elapsed, unit="s")
        + pd.to_timedelta(VOLTAGE_TIME_OFFSET_SECONDS, unit="s")
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"Voltage hour rollovers detected: {wraps}")

    return df[["timestamp", "Reading"]].rename(
        columns={"Reading": "Measured_Voltage_millivolts"}
    )


def read_conductivity_data(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(*prefixes):
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

    df["timestamp"] = pd.to_datetime(
        df[datetime_col].astype(str).str.strip(),
        errors="coerce"
    )

    df["MeasuredSurfaceWaterTemperatureCelsius"] = pd.to_numeric(
        df[temperature_col], errors="coerce"
    )
    df["MeasuredSurfaceWaterConductivitymicrosiemenspercm"] = pd.to_numeric(
        df[spcond_col], errors="coerce"
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "MeasuredSurfaceWaterTemperatureCelsius",
            "MeasuredSurfaceWaterConductivitymicrosiemenspercm",
        ]
    ).copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df[
        [
            "timestamp",
            "MeasuredSurfaceWaterTemperatureCelsius",
            "MeasuredSurfaceWaterConductivitymicrosiemenspercm",
        ]
    ]


def format_measurement_time(ts):
    return f"{ts.hour}:{ts.minute:02d}:{ts.second:02d}"


def assign_eddy_sequence_labels(df):
    """
    For rows already filtered to mode_flag == 'eddy', assign labels based on
    contiguous eddy sequences in the merged dataframe order.
    """
    df = df.copy()

    df["row_gap"] = df.index.to_series().diff().fillna(1)
    df["eddy_sequence_id"] = (df["row_gap"] != 1).cumsum()
    df["DRIFT_MEASUREMENT_LOCATION"] = "eddy_" + df["eddy_sequence_id"].astype(int).astype(str)

    return df.drop(columns=["row_gap"])


def print_time_ranges(gnss_df, voltage_df, conductivity_df):
    print("\n--- Time range diagnostics (local) ---")
    for name, d in [
        ("GNSS", gnss_df),
        ("Voltage", voltage_df),
        ("Conductivity", conductivity_df),
    ]:
        if d.empty:
            print(f"{name:13s} EMPTY")
            continue
        t0 = d["timestamp"].min()
        t1 = d["timestamp"].max()
        span_min = (t1 - t0).total_seconds() / 60.0
        print(f"{name:13s} {t0}  ->  {t1}   ({len(d)} rows, {span_min:.1f} min)")

    if not gnss_df.empty and not voltage_df.empty:
        overlap_start = max(gnss_df["timestamp"].min(), voltage_df["timestamp"].min())
        overlap_end = min(gnss_df["timestamp"].max(), voltage_df["timestamp"].max())
        if overlap_end <= overlap_start:
            gap_min = (overlap_start - overlap_end).total_seconds() / 60.0
            print(
                f"WARNING: GNSS and Voltage windows DO NOT OVERLAP "
                f"(separated by {gap_min:.1f} min). "
                f"Check VOLTAGE_START_LOCAL_DATETIME."
            )
        else:
            print(
                f"GNSS/Voltage overlap: {overlap_start} -> {overlap_end} "
                f"({(overlap_end - overlap_start).total_seconds() / 60.0:.1f} min)"
            )
    print("--------------------------------------\n")


def main():
    gnss_df = read_arrow_data(ARROW_FILE)
    voltage_df = read_voltage_data(VOLTAGE_FILE)
    conductivity_df = read_conductivity_data(CONDUCTIVITY_FILE)

    gnss_df = gnss_df.sort_values("timestamp").reset_index(drop=True)
    voltage_df = voltage_df.sort_values("timestamp").reset_index(drop=True)
    conductivity_df = conductivity_df.sort_values("timestamp").reset_index(drop=True)

    print_time_ranges(gnss_df, voltage_df, conductivity_df)

    # -------------------------------------------------------------------------
    # SP / Voltage merge (nearest GNSS within +/- 2 sec)
    # -------------------------------------------------------------------------
    merged = pd.merge_asof(
        voltage_df,
        gnss_df,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MAX_TIME_DIFF_SECONDS),
    )

    merged = merged.dropna(subset=["Longitude", "Latitude", "mode_flag"]).copy()

    merged["Segment_ID"] = SEGMENT_ID
    merged["Segment_Start_Location"] = SEGMENT_START_LOCATION
    merged["Segment_End_Location"] = SEGMENT_END_LOCATION
    merged["Measurement_Time"] = merged["timestamp"].apply(format_measurement_time)

    if not merged.empty:
        print("mode_flag counts in matched rows:")
        print(merged["mode_flag"].value_counts().to_string())

    # -------------------------------------------------------------------------
    # SP output: ONLY mode_flag == 'drift'
    # -------------------------------------------------------------------------
    sp_df = merged[merged["mode_flag"] == "drift"].copy()
    sp_df["Measurement_Number"] = range(1, len(sp_df) + 1)

    sp_output_df = sp_df[
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
    ].copy()

    sp_output_df.to_csv(SP_OUTPUT_FILE, index=False)

    # -------------------------------------------------------------------------
    # VDRIFT output: ONLY mode_flag == 'eddy'
    # Each contiguous eddy sequence gets DriftMeasurementLocation = eddy_1, eddy_2, ...
    # -------------------------------------------------------------------------
    eddy_df = merged[merged["mode_flag"] == "eddy"].copy()

    if not eddy_df.empty:
        eddy_df = assign_eddy_sequence_labels(eddy_df)
        eddy_df["Measurement_Number"] = eddy_df.groupby("eddy_sequence_id").cumcount() + 1
    else:
        eddy_df["DRIFT_MEASUREMENT_LOCATION"] = pd.Series(dtype="object")
        eddy_df["eddy_sequence_id"] = pd.Series(dtype="int")
        eddy_df["Measurement_Number"] = pd.Series(dtype="int")

    vdrift_output_df = eddy_df.rename(
        columns={
            "Segment_ID": "SegmentID",
            "Measurement_Number": "MeasurementNumber",
            "DRIFT_MEASUREMENT_LOCATION": "DriftMeasurementLocation",
            "Measurement_Time": "MeasurementTime",
            "Measured_Voltage_millivolts": "MeasuredVoltagemillivolts",
        }
    )[
        [
            "SegmentID",
            "MeasurementNumber",
            "DriftMeasurementLocation",
            "MeasurementTime",
            "MeasuredVoltagemillivolts",
        ]
    ].copy()

    vdrift_output_df.to_csv(VDRIFT_OUTPUT_FILE, index=False)

    # -------------------------------------------------------------------------
    # Conductivity output: nearest GNSS within hard-coded +/- 5 sec
    # -------------------------------------------------------------------------
    conductivity_merged = pd.merge_asof(
        conductivity_df,
        gnss_df[["timestamp", "Longitude", "Latitude"]],
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=5),
    )

    conductivity_merged = conductivity_merged.dropna(
        subset=["Longitude", "Latitude"]
    ).copy()

    conductivity_merged["SegmentID"] = SEGMENT_ID
    conductivity_merged["SegmentStartLocation"] = SEGMENT_START_LOCATION
    conductivity_merged["SegmentEndLocation"] = SEGMENT_END_LOCATION
    conductivity_merged["MeasurementNumber"] = range(1, len(conductivity_merged) + 1)
    conductivity_merged["MeasurementTime"] = conductivity_merged["timestamp"].apply(
        format_measurement_time
    )

    conductivity_output_df = conductivity_merged[
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
    ].copy()

    conductivity_output_df.to_csv(CONDUCTIVITY_OUTPUT_FILE, index=False)

    print("\nDone.")
    print(f"Arrow time mode: {ARROW_TIME_MODE}")
    print(f"GNSS rows: {len(gnss_df)}")
    print(f"Voltage rows: {len(voltage_df)}")
    print(f"Matched voltage/SP rows: {len(merged)}")
    print(f"SP rows written (drift): {len(sp_output_df)}")
    print(f"VDRIFT rows written (eddy): {len(vdrift_output_df)}")
    print(f"Conductivity rows: {len(conductivity_df)}")
    print(f"Matched conductivity rows (+/- 5 s): {len(conductivity_output_df)}")
    print(f"SP output: {SP_OUTPUT_FILE}")
    print(f"VDRIFT output: {VDRIFT_OUTPUT_FILE}")
    print(f"Conductivity output: {CONDUCTIVITY_OUTPUT_FILE}")


if __name__ == "__main__":
    main()