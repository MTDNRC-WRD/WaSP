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
- Prompts for a COM port at startup (default COM11), with retry on failure
- Interactive keyboard controls:
    - A/a: set mode to "anomaly" (also resumes if paused)
    - E/e: set mode to "eddy" (also resumes if paused)
    - D/d: set mode to "drift" (also resumes if paused)
    - P/p: pause logging
    - R/r: resume logging with the last-used tag
    - S/s: stop logging and exit the script

The current mode is written to a new CSV column "mode_flag" for each row.
"""

import csv
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pynmea2
import serial
import winsound
import msvcrt  # Windows-only keyboard input [web:44]


DEFAULT_PORT_NUM = 11
BAUD = 9600
OUT_CSV = "arrow_log"
LOG_INTERVAL_SEC = 1.0
SERIAL_TIMEOUT_SEC = 2

FAIL_TXT = "arrow_failure_notice.txt"
ERROR_LOG = "arrow_error_log.txt"

# Modes and recording states
MODE_NONE = ""
MODE_ANOMALY = "anomaly"
MODE_EDDY = "eddy"
MODE_DRIFT = "drift"

STATE_RUNNING = "running"
STATE_PAUSED = "paused"
STATE_STOPPED = "stopped"

# Global PORT is resolved at startup and used in failure notices
PORT = None


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


def local_now_iso():
    """Return current local PC time in ISO format with local timezone offset."""
    return datetime.now().astimezone().isoformat()


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


def prompt_com_port(default_num=DEFAULT_PORT_NUM):
    """Ask the user for a COM port number, defaulting if blank.

    Returns a string like "COM11".
    """
    raw = input(
        f"Enter COM port number [default {default_num}]: "
    ).strip()

    if raw == "":
        port_num = default_num
    else:
        # Allow either "9" or "COM9" style input
        raw_upper = raw.upper()
        if raw_upper.startswith("COM"):
            raw_upper = raw_upper[3:]
        try:
            port_num = int(raw_upper)
        except ValueError:
            print(f"[WARN] Could not parse '{raw}' as a port number, using default {default_num}.")
            port_num = default_num

    return f"COM{port_num}"


def open_serial_with_retry(baud, timeout_sec):
    """Prompt for a COM port, try to open it, and retry on failure.

    Returns an open serial.Serial instance and sets the global PORT.
    """
    global PORT

    port_name = prompt_com_port(DEFAULT_PORT_NUM)

    while True:
        try:
            connection = serial.Serial(port_name, baud, timeout=timeout_sec)
            PORT = port_name
            print(f"[STATUS] Connected on {port_name}.")
            return connection
        except serial.SerialException as exc:
            print(f"[ERROR] Could not open {port_name}: {exc}")
            port_name = prompt_com_port(DEFAULT_PORT_NUM)


def print_instructions(output_csv_path):
    """Print startup instructions and status to the terminal."""
    print("=" * 80)
    print("Arrow GNSS Logger with Interactive Flags")
    print(f"Port       : {PORT}")
    print(f"Baud       : {BAUD}")
    print(f"CSV output : {output_csv_path}")
    print(f"Interval   : {LOG_INTERVAL_SEC} sec")
    print("-" * 80)
    print("Keyboard controls (case-insensitive):")
    print("  A : set mode to 'anomaly' (resumes if paused)")
    print("  E : set mode to 'eddy' (resumes if paused)")
    print("  D : set mode to 'drift' (resumes if paused)")
    print("  P : pause logging (no new rows written)")
    print("  R : resume logging with the last-used tag")
    print("  S : stop logging and exit")
    print("-" * 80)
    print("Press Ctrl+C for emergency stop.")
    print("=" * 80)


def poll_keyboard(current_mode, recording_state):
    """Check for key presses and update mode / recording state.

    Returns updated (current_mode, recording_state).
    Keys (case-insensitive):
      A: anomaly mode (resumes if paused)
      E: eddy mode (resumes if paused)
      D: drift mode (resumes if paused)
      P: pause logging
      R: resume logging with last-used tag
      S: stop logging and exit
    """
    # Non-blocking keyboard check on Windows [web:44][web:51]
    while msvcrt.kbhit():
        ch = msvcrt.getch()
        try:
            key = ch.decode("utf-8")
        except Exception:
            continue

        key = key.upper()
        was_paused = recording_state == STATE_PAUSED

        if key == "A":
            current_mode = MODE_ANOMALY
            recording_state = STATE_RUNNING
            if was_paused:
                print("[STATUS] Resumed. Mode set to ANOMALY.")
            else:
                print("[STATUS] Mode set to ANOMALY.")
        elif key == "E":
            current_mode = MODE_EDDY
            recording_state = STATE_RUNNING
            if was_paused:
                print("[STATUS] Resumed. Mode set to EDDY.")
            else:
                print("[STATUS] Mode set to EDDY.")
        elif key == "D":
            current_mode = MODE_DRIFT
            recording_state = STATE_RUNNING
            if was_paused:
                print("[STATUS] Resumed. Mode set to DRIFT.")
            else:
                print("[STATUS] Mode set to DRIFT.")
        elif key == "P":
            recording_state = STATE_PAUSED
            print("[STATUS] Recording PAUSED.")
        elif key == "R":
            recording_state = STATE_RUNNING
            print(f"[STATUS] Recording RESUMED with mode={current_mode or 'none'}.")
        elif key == "S":
            recording_state = STATE_STOPPED
            print("[STATUS] Recording STOPPED. Exiting.")

    return current_mode, recording_state


def print_row_status(local_time_str, mode_flag, gps_qual, num_sats):
    """Print a brief status line for each logged row."""
    print(
        f"[ROW] {local_time_str} | "
        f"mode={mode_flag or 'none'} | "
        f"fix={gps_qual} | sats={num_sats}"
    )


def main():
    """Read GNSS NMEA sentences from serial and log selected fields to CSV."""
    latest_pdop = None
    latest_hdop = None
    latest_vdop = None
    last_log_time = 0.0
    connection_lost = False
    current_mode = MODE_NONE
    recording_state = STATE_RUNNING
    was_paused_last_loop = False

    output_csv_path = resolve_output_csv_path(OUT_CSV)

    serial_connection = open_serial_with_retry(BAUD, SERIAL_TIMEOUT_SEC)

    with serial_connection, open(
        output_csv_path, "w", newline="", encoding="utf-8"
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            [
                "pc_local_time",
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
                "mode_flag",     # new column for drift/eddy/anomaly modes
                "raw_sentence",
            ]
        )
        output_file.flush()

        print_instructions(output_csv_path)

        while True:
            # Check keyboard first so pause/resume/stop applies quickly
            current_mode, recording_state = poll_keyboard(current_mode, recording_state)

            if recording_state == STATE_STOPPED:
                print("[STATUS] Exiting logging loop due to STOP command.")
                break

            if recording_state == STATE_PAUSED:
                if not was_paused_last_loop:
                    print("[STATUS] Recording paused.")
                was_paused_last_loop = True
            else:
                was_paused_last_loop = False

            raw_bytes = serial_connection.readline()

            if not raw_bytes:
                if not connection_lost:
                    print("[WARN] Connection lost: no serial data received.")
                    connection_lost = True
                time.sleep(0.05)
                continue

            if connection_lost:
                print("[STATUS] Connection restored.")
                connection_lost = False

            line = raw_bytes.decode("ascii", errors="ignore").strip()

            if not line.startswith("$"):
                continue

            try:
                message = pynmea2.parse(line)
            except pynmea2.ParseError:
                continue

            # Update DOP from GSA
            if message.sentence_type == "GSA":
                latest_pdop = float(message.pdop) if message.pdop else None
                latest_hdop = float(message.hdop) if message.hdop else None
                latest_vdop = float(message.vdop) if message.vdop else None

            # Log position from GGA
            elif message.sentence_type == "GGA":
                current_time = time.time()

                if current_time - last_log_time < LOG_INTERVAL_SEC:
                    continue

                # If paused, skip writing but keep timing cadence (no per-row print)
                if recording_state != STATE_RUNNING:
                    last_log_time = current_time
                    continue

                latitude_dd = dm_to_decimal(message.lat, message.lat_dir)
                longitude_dd = dm_to_decimal(message.lon, message.lon_dir)

                local_time_str = local_now_iso()
                utc_time_str = utc_now_iso()

                writer.writerow(
                    [
                        local_time_str,
                        utc_time_str,
                        str(message.timestamp) if message.timestamp else "",
                        latitude_dd,
                        longitude_dd,
                        message.altitude,
                        message.gps_qual,
                        message.num_sats,
                        latest_pdop,
                        (
                            float(message.horizontal_dil)
                            if getattr(message, "horizontal_dil", None)
                            else latest_hdop
                        ),
                        latest_vdop,
                        current_mode,  # persistent flag: drift/eddy/anomaly/none
                        line,
                    ]
                )
                output_file.flush()
                last_log_time = current_time

                # Print a short per-row status line
                print_row_status(
                    local_time_str,
                    current_mode,
                    message.gps_qual,
                    message.num_sats,
                )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STATUS] Logging stopped by user (Ctrl+C).")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception:
        error_text = traceback.format_exc()
        write_failure_notice(error_text)
        play_failure_sound()
        print("\n[ERROR] Logger crashed. See arrow_failure_notice.txt and arrow_error_log.txt")
        raise
