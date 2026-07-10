import pandas as pd


# =============================================================================
# USER SETTINGS - EDIT THESE
# =============================================================================

ARROW_FILE = r"C:\Users\CND367\Documents\WaSP\test1\fake_arrow_log.csv"
VOLTAGE_FILE = r"C:\Users\CND367\Documents\WaSP\test1\voltage_log.csv"
OUTPUT_FILE = r"C:\Users\CND367\Documents\WaSP\test1\formatted_self_potential.csv"

# Choose which Arrow timestamp column to use:
#   "utc"   -> use pc_utc_time and convert to local time
#   "local" -> use pc_local_time directly
ARROW_TIME_MODE = "utc"

TIMEZONE = "MDT"   # only used when ARROW_TIME_MODE = "utc"
UTC_TO_LOCAL_HOURS = -6 if TIMEZONE == "MDT" else -7

# The voltage file uses shorthand elapsed times like 25:32.7, 55:04.4, etc.
# Set this to the LOCAL datetime that corresponds to the FIRST voltage record.
VOLTAGE_START_LOCAL_DATETIME = "2026-07-02 11:25:32.7"

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

    required = ["lat_dd", "lon_dd"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Arrow file is missing required columns: {missing}")

    df = df.copy()

    if ARROW_TIME_MODE.lower() == "local":
        if "pc_local_time" not in df.columns:
            raise ValueError(
                "ARROW_TIME_MODE is 'local' but pc_local_time was not found in the Arrow file."
            )

        df["timestamp"] = pd.to_datetime(
            df["pc_local_time"].astype(str).str.strip(),
            errors="coerce"
        )

        df = df.dropna(subset=["timestamp", "lat_dd", "lon_dd"]).copy()

        if pd.api.types.is_datetime64tz_dtype(df["timestamp"]):
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

        df = df.dropna(subset=["timestamp_utc", "lat_dd", "lon_dd"]).copy()

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

    return df[["timestamp", "lon_dd", "lat_dd"]].rename(
        columns={
            "lon_dd": "Longitude",
            "lat_dd": "Latitude",
        }
    )


def parse_elapsed_mmss(value):
    """
    Parse shorthand elapsed time strings like:
      25:32.7  -> 25 min 32.7 sec
      55:04.4  -> 55 min 4.4 sec
      16:33.4  -> 16 min 33.4 sec
    Returns elapsed seconds as float.
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
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = ["Time", "Reading"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Voltage file is missing required columns: {missing}")

    df = df.copy()

    df["elapsed_seconds"] = df["Time"].apply(parse_elapsed_mmss)
    df["Reading"] = pd.to_numeric(df["Reading"], errors="coerce")

    df = df.dropna(subset=["elapsed_seconds", "Reading"]).copy()
    df = df.sort_values("elapsed_seconds").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid voltage time/reading rows found.")

    start_local = pd.to_datetime(VOLTAGE_START_LOCAL_DATETIME, errors="raise")

    first_elapsed = df["elapsed_seconds"].iloc[0]
    df["timestamp"] = (
        start_local
        + pd.to_timedelta(df["elapsed_seconds"] - first_elapsed, unit="s")
        + pd.to_timedelta(VOLTAGE_TIME_OFFSET_SECONDS, unit="s")
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("datetime64[ns]")

    return df[["timestamp", "Reading"]].rename(
        columns={"Reading": "Measured_Voltage_millivolts"}
    )


def format_measurement_time(ts):
    return f"{ts.hour}:{ts.minute:02d}:{ts.second:02d}"


def main():
    gnss_df = read_arrow_data(ARROW_FILE)
    voltage_df = read_voltage_data(VOLTAGE_FILE)

    gnss_df = gnss_df.sort_values("timestamp").reset_index(drop=True)
    voltage_df = voltage_df.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        voltage_df,
        gnss_df,
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MAX_TIME_DIFF_SECONDS),
    )

    merged = merged.dropna(subset=["Longitude", "Latitude"]).copy()

    merged["Segment_ID"] = SEGMENT_ID
    merged["Segment_Start_Location"] = SEGMENT_START_LOCATION
    merged["Segment_End_Location"] = SEGMENT_END_LOCATION
    merged["Measurement_Number"] = range(1, len(merged) + 1)
    merged["Measurement_Time"] = merged["timestamp"].apply(format_measurement_time)

    output_df = merged[
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

    output_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Done. Wrote {len(output_df)} rows to:")
    print(OUTPUT_FILE)
    print(f"Arrow time mode: {ARROW_TIME_MODE}")
    print(f"GNSS rows: {len(gnss_df)}")
    print(f"Voltage rows: {len(voltage_df)}")
    print(f"Matched rows: {len(output_df)}")
    print(f"GNSS dtype: {gnss_df['timestamp'].dtype}")
    print(f"Voltage dtype: {voltage_df['timestamp'].dtype}")


if __name__ == "__main__":
    main()