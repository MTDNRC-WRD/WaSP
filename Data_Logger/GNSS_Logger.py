import csv
from datetime import datetime, timezone

import csv
import time
from datetime import datetime, timezone
import serial
import pynmea2

PORT = "COM5"
BAUD = 9600
OUT_CSV = "arrow_log.csv"
LOG_INTERVAL_SEC = 5.0   # write one row every 5 seconds

def dm_to_decimal(value, direction):
    if value in (None, ""):
        return None
    v = float(value)
    degrees = int(v / 100)
    minutes = v - degrees * 100
    dec = degrees + minutes / 60.0
    if direction in ("S", "W"):
        dec *= -1
    return dec

latest_pdop = None
latest_hdop = None
latest_vdop = None
last_log_time = 0.0

try:
    with serial.Serial(PORT, BAUD, timeout=2) as ser, open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
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
            "raw_sentence"
        ])

        print(f"Logging started on {PORT}. Interval = {LOG_INTERVAL_SEC} sec. Press Ctrl+C to stop.")

        while True:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue

            try:
                msg = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue

            if msg.sentence_type == "GSA":
                latest_pdop = float(msg.pdop) if msg.pdop else None
                latest_hdop = float(msg.hdop) if msg.hdop else None
                latest_vdop = float(msg.vdop) if msg.vdop else None

            elif msg.sentence_type == "GGA":
                now = time.time()

                if now - last_log_time < LOG_INTERVAL_SEC:
                    continue

                lat = dm_to_decimal(msg.lat, msg.lat_dir)
                lon = dm_to_decimal(msg.lon, msg.lon_dir)

                writer.writerow([
                    datetime.now(timezone.utc).isoformat(),
                    str(msg.timestamp) if msg.timestamp else "",
                    lat,
                    lon,
                    msg.altitude,
                    msg.gps_qual,
                    msg.num_sats,
                    float(msg.horizontal_dil) if msg.horizontal_dil else latest_hdop,
                    latest_pdop,
                    latest_vdop,
                    line
                ])
                f.flush()
                last_log_time = now

except KeyboardInterrupt:
    print("\nLogging stopped by user.")