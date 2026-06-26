"""Log Arrow GNSS NMEA data to CSV at a fixed interval.

This script reads NMEA sentences from a serial-connected GNSS receiver,
tracks the most recent dilution of precision (DOP) values from GSA messages,
and writes position records from GGA messages to a CSV file.

The output CSV includes:
- PC UTC timestamp
- NMEA timestamp
- Latitude and longitude in decimal degrees
- Altitude in meters
- Fix quality
- Number of satellites
- PDOP, HDOP, and VDOP
- The raw NMEA sentence

The script runs until interrupted by the user.
"""

import csv
import time
from datetime import datetime, timezone

import pynmea2
import serial


PORT = "COM5"
BAUD = 9600
OUT_CSV = "arrow_log.csv"
LOG_INTERVAL_SEC = 1.0


def dm_to_decimal(value, direction):
    """Convert NMEA degrees-minutes coordinates to decimal degrees.

    Args:
        value: Coordinate value in NMEA degrees-minutes format, such as
            "4530.1234". May be None or an empty string.
        direction: Cardinal direction indicator, usually one of
            "N", "S", "E", or "W".

    Returns:
        The coordinate in decimal degrees as a float, or None if the input
        value is empty.
    """
    if value in (None, ""):
        return None

    numeric_value = float(value)
    degrees = int(numeric_value / 100)
    minutes = numeric_value - degrees * 100
    decimal_degrees = degrees + minutes / 60.0

    if direction in ("S", "W"):
        decimal_degrees *= -1

    return decimal_degrees


def main():
    """Read GNSS NMEA sentences from serial and log selected fields to CSV.

    The function listens for GSA and GGA messages from the configured serial
    port. GSA messages update the most recent DOP values, and GGA messages
    are written to the CSV file no more frequently than LOG_INTERVAL_SEC.

    Raises:
        serial.SerialException: If the serial port cannot be opened or read.
        OSError: If the output CSV file cannot be created or written.
    """
    latest_pdop = None
    latest_hdop = None
    latest_vdop = None
    last_log_time = 0.0

    with serial.Serial(PORT, BAUD, timeout=2) as serial_connection, open(
        OUT_CSV, "w", newline=""
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

        print(
            f"Logging started on {PORT}. "
            f"Interval = {LOG_INTERVAL_SEC} sec. "
            "Press Ctrl+C to stop."
        )

        while True:
            line = serial_connection.readline().decode(
                "ascii", errors="ignore"
            ).strip()

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
                        datetime.now(timezone.utc).isoformat(),
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