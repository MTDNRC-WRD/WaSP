"""Log Arrow GNSS NMEA data to CSV at a fixed interval.

This script reads NMEA sentences from a serial-connected GNSS receiver,
tracks the most recent dilution of precision (DOP) values from GSA messages,
and writes position records from GGA messages to a CSV file.

Features:
- Logs GGA position records to CSV
- Tracks latest PDOP, HDOP, and VDOP from GSA messages
- Flushes CSV after each write so data is preserved during interruptions
- Prints terminal status messages for connection loss/restoration
- Avoids overwriting an existing CSV by adding a suffix like _01, _02, ...
- Writes a failure notice and traceback log if the script crashes
- Plays a Windows notification sound on crash
"""

import csv
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pynmea2
import serial
import winsound


PORT = "COM5"
BAUD = 9600
OUT_CSV = "arrow_log.csv"
LOG_INTERVAL_SEC = 1.0
SERIAL_TIMEOUT_SEC = 2

FAIL_TXT = "arrow_failure_notice.txt"
ERROR_LOG = "arrow_error_log.txt"


def dm_to_decimal(value, direction):
    """Convert NMEA degrees-minutes coordinates to decimal degrees."""
    if value in (None, ""):
        return None

    numeric_value = float(value)
    degrees = int(numeric_value / 100)
    minutes = numeric_value - degrees * 100
    decimal_degrees = degrees + minutes / 60.0

    if direction in ("S", "W"):
        decimal_degrees *= -1

    return decimal_degrees


def utc_now_iso():
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def resolve_output_csv_path(base_name):
    """Return a dated, non-overwriting CSV path.

    Examples:
        arrow_log_20260707.csv
        arrow_log_20260707_01.csv
        arrow_log_20260707_02.csv
    """
    today_str = datetime.now().strftime("%Y%m%d")
    base_path = Path(f"{base_name}_{today_str}.csv")

    if not base_path.exists():
        return base_path

    index = 1
    while True:
        candidate = Path(f"{base_name}_{today_str}_{index:02d}.csv")
        if not candidate.exists():
            return candidate
        index += 1


def play_failure_sound():
    """Play a Windows alert sound on crash."""
    try:
        for frequency in (1400, 1800, 1400):
            winsound.Beep(frequency, 350)
            time.sleep(0.15)
    except Exception:
        try:
            winsound.MessageBeep(winsound.MB_ICONHAND)
        except Exception:
            pass


def write_failure_notice(error_text):
    """Write the latest crash to a text file and append it to a log."""
    timestamp = utc_now_iso()

    notice = (
        "ARROW LOGGER FAILURE\n"
        f"UTC time: {timestamp}\n"
        f"Port: {PORT}\n"
        f"Baud: {BAUD}\n\n"
        "Traceback:\n"
        f"{error_text}\n"
    )

    Path(FAIL_TXT).write_text(notice, encoding="utf-8")

    with open(ERROR_LOG, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(notice)


def main():
    """Read GNSS NMEA sentences from serial and log selected fields to CSV."""
    latest_pdop = None
    latest_hdop = None
    latest_vdop = None
    last_log_time = 0.0
    connection_lost = False
    output_csv_path = resolve_output_csv_path(OUT_CSV)

    with serial.Serial(PORT, BAUD, timeout=SERIAL_TIMEOUT_SEC) as serial_connection, open(
        output_csv_path, "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
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
        )
        output_file.flush()

        print(
            f"Logging started on {PORT}. "
            f"Writing to {output_csv_path}. "
            f"Interval = {LOG_INTERVAL_SEC} sec. "
            "Press Ctrl+C to stop."
        )

        while True:
            raw_bytes = serial_connection.readline()

            if not raw_bytes:
                if not connection_lost:
                    print("Connection lost: no serial data received.")
                    connection_lost = True
                continue

            if connection_lost:
                print("Connection restored.")
                connection_lost = False

            line = raw_bytes.decode("ascii", errors="ignore").strip()

            if not line.startswith("$"):
                continue

            try:
                message = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue

            if message.sentence_type == "GSA":
                latest_pdop = float(message.pdop) if message.pdop else None
                latest_hdop = float(message.hdop) if message.hdop else None
                latest_vdop = float(message.vdop) if message.vdop else None

            elif message.sentence_type == "GGA":
                current_time = time.time()

                if current_time - last_log_time < LOG_INTERVAL_SEC:
                    continue

                latitude_dd = dm_to_decimal(message.lat, message.lat_dir)
                longitude_dd = dm_to_decimal(message.lon, message.lon_dir)

                writer.writerow(
                    [
                        utc_now_iso(),
                        str(message.timestamp) if message.timestamp else "",
                        latitude_dd,
                        longitude_dd,
                        message.altitude,
                        message.gps_qual,
                        message.num_sats,
                        (
                            float(message.horizontal_dil)
                            if message.horizontal_dil
                            else latest_hdop
                        ),
                        latest_pdop,
                        latest_vdop,
                        line,
                    ]
                )
                output_file.flush()
                last_log_time = current_time


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nLogging stopped by user.")
    except Exception:
        error_text = traceback.format_exc()
        write_failure_notice(error_text)
        play_failure_sound()
        print("\nLogger crashed. See arrow_failure_notice.txt and arrow_error_log.txt")
        raise