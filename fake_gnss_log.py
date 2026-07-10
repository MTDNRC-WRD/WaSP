import pandas as pd
import math


# =============================================================================
# USER SETTINGS
# =============================================================================

VOLTAGE_FILE = r"C:\Users\CND367\Documents\WaSP\test1\voltage_log.csv"
OUTPUT_ARROW_FILE = r"C:\Users\CND367\Documents\WaSP\test1\fake_arrow_log.csv"

# Start datetime for the FIRST voltage record, in local time
# Example: if record 1 ("25:32.7") happened at 2026-07-02 11:25:32.7 local,
# set that here.
START_LOCAL_DATETIME = "2026-07-02 11:25:32.7"

TIMEZONE = "MDT"   # "MDT" or "MST"
LOCAL_TO_UTC_HOURS = 6 if TIMEZONE == "MDT" else 7

START_LAT_DD = 46.611170
START_LON_DD = -111.894030
START_ALT_M = 1162.75

FEET_EAST_PER_POINT = 2.0


# =============================================================================
# FUNCTIONS
# =============================================================================

def read_voltage_file(path):
    df = pd.read_csv(path, encoding="utf-8-sig")

    required = ["Time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Voltage file missing required columns: {missing}")

    return df.copy()


def parse_elapsed_time_to_seconds(time_str):
    """
    Convert strings like:
      25:32.7  -> 25 min, 32.7 sec
      54:58.4  -> 54 min, 58.4 sec
      16:33.4  -> 16 min, 33.4 sec
    into total seconds.
    """
    s = str(time_str).strip()

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


def degrees_lon_per_foot(lat_dd):
    meters_per_degree_lon = 111320.0 * math.cos(math.radians(lat_dd))
    feet_per_degree_lon = meters_per_degree_lon * 3.28084
    return 1.0 / feet_per_degree_lon


def make_fake_raw_sentence(utc_ts, lat_dd, lon_dd, alt_m, fix_quality=5, num_sats=19, hdop=0.8):
    hhmmss = utc_ts.strftime("%H%M%S.%f")[:9]

    lat_abs = abs(lat_dd)
    lat_deg = int(lat_abs)
    lat_min = (lat_abs - lat_deg) * 60
    lat_nmea = f"{lat_deg:02d}{lat_min:011.8f}"
    lat_hemi = "N" if lat_dd >= 0 else "S"

    lon_abs = abs(lon_dd)
    lon_deg = int(lon_abs)
    lon_min = (lon_abs - lon_deg) * 60
    lon_nmea = f"{lon_deg:03d}{lon_min:011.8f}"
    lon_hemi = "E" if lon_dd >= 0 else "W"

    return (
        f"$GPGGA,{hhmmss},{lat_nmea},{lat_hemi},{lon_nmea},{lon_hemi},"
        f"{fix_quality},{num_sats},{hdop},{alt_m:.3f},M,-17.393,M,,"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = read_voltage_file(VOLTAGE_FILE)

    df["elapsed_seconds"] = df["Time"].apply(parse_elapsed_time_to_seconds)
    df = df.dropna(subset=["elapsed_seconds"]).copy()

    if df.empty:
        print("No valid elapsed time records found in voltage file.")
        return

    start_local = pd.to_datetime(START_LOCAL_DATETIME, errors="raise")

    first_elapsed = df["elapsed_seconds"].iloc[0]
    df["timestamp_local"] = start_local + pd.to_timedelta(df["elapsed_seconds"] - first_elapsed, unit="s")

    df["timestamp_utc"] = df["timestamp_local"] + pd.to_timedelta(LOCAL_TO_UTC_HOURS, unit="h")
    df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")

    deg_lon_per_ft = degrees_lon_per_foot(START_LAT_DD)

    df["pc_utc_time"] = df["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    df["nmea_time"] = df["timestamp_utc"].dt.strftime("%H:%M:%S+00:00")

    df["lat_dd"] = START_LAT_DD
    df["lon_dd"] = [
        START_LON_DD + (i * FEET_EAST_PER_POINT * deg_lon_per_ft)
        for i in range(len(df))
    ]

    df["alt_m"] = START_ALT_M
    df["fix_quality"] = 5
    df["num_sats"] = 19
    df["pdop"] = 0.4
    df["hdop"] = 0.8
    df["vdop"] = 0.7

    df["raw_sentence"] = [
        make_fake_raw_sentence(ts, lat, lon, START_ALT_M)
        for ts, lat, lon in zip(df["timestamp_utc"], df["lat_dd"], df["lon_dd"])
    ]

    arrow_df = df[
        [
            "pc_utc_time",
            "nmea_time",
            "lat_dd",
            "lon_dd",
            "alt_m",
            "fix_quality",
            "num_sats",
            "pdop",
            "hdop",
            "vdop",
            "raw_sentence",
        ]
    ].copy()

    arrow_df.to_csv(OUTPUT_ARROW_FILE, index=False)

    print(f"Done. Wrote {len(arrow_df)} rows to:")
    print(OUTPUT_ARROW_FILE)
    print(f"First local timestamp: {df['timestamp_local'].min()}")
    print(f"Last local timestamp:  {df['timestamp_local'].max()}")


if __name__ == "__main__":
    main()