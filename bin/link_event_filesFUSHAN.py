#!/usr/bin/env python

# (Hiroto, 2025/10/30) ver2.0: [Revise] Improved version with argparse, structured logging, --shift-max, --tstart, --tend and dry-run/debug support for operational use.
# (Hiroto, 2025/12/11) ver2.1: [Revise] Add an --src option to select a specific astronomical sources (e.g. b0329, crab)

import os
import sys
import pandas as pd
from glob import glob
from datetime import datetime, timedelta
import argparse

# ============================================================
# Logging Utility
# ============================================================
def print_log(level: str, msg: str):
    """Unified logging with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper():7}] {msg}")

# ============================================================
# Command line argument parser
# ============================================================
parser = argparse.ArgumentParser(
    prog="link_event_files.py",
    usage="%(prog)s <input_csv> [--search-dir PATH] [--tstart TSTART] [--tend TEND] [--shift-max N] [--src TARGET] [--debug] [--dry-run]",
    description=(
        "Link event-related raw data files based on EventID timestamps.\n"
        "For each event, this script searches for matching data files "
        "in mounted disks and creates symbolic links in a directory named after the EventID."
    ),
    epilog=(
        "Example:\n"
        "  python link_event_files.py events.csv\n"
        "  python link_event_files.py events.csv --tstart '2025-10-01 00:00:00' --tend '2025-10-01 01:00:00'\n"
        "  python link_event_files.py events.csv --search-dir /burstt01/disk2/data --shift-max 10 --src 'b0329' --dry-run\n"
    ),
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("input_csv", help="CSV file containing the event list (must include 'EventID').")
parser.add_argument("--search-dir", type=str, default=None,
                    help="Root directory to search (default: auto-detect /burstt*/disk*/data).")
parser.add_argument("--tstart", type=str, default=None,
                    help="Start time in format 'YYYY-MM-DD HH:MM:SS' (UTC)")
parser.add_argument("--tend", type=str, default=None,
                    help="End time in format 'YYYY-MM-DD HH:MM:SS' (UTC)")
parser.add_argument("--shift-max", type=int, default=5,
                    help="Maximum seconds to shift when searching (default: 5).")
parser.add_argument("--src", type=str, default=None,
                    help="Specify an astronomical source")
parser.add_argument("--debug", action="store_true",
                    help="Enable verbose debug output for troubleshooting.")
parser.add_argument("--dry-run", action="store_true",
                    help="Simulate all operations without creating directories or symlinks.")

# Show usage if no options
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

# ============================================================
# Input and output file setup
# ============================================================
input_file = args.input_csv
if not os.path.exists(input_file):
    print_log("error", f"Input file not found: {input_file}")
    sys.exit(1)

base, ext = os.path.splitext(input_file)
output_file = f"{base}_linked_files{ext}"

# ============================================================
# Detect search directories
# ============================================================
if args.search_dir:
    search_dirs = [args.search_dir]
else:
    search_dirs = sorted(glob("/burstt*/disk*/data"))

if not search_dirs:
    print_log("error", "No search directories found. Use --search-dir to specify.")
    sys.exit(1)

print_log("info", f"Found {len(search_dirs)} search directories:")
for d in search_dirs:
    print(f"  - {d}")

# ============================================================
# Load input CSV + Select specific date range
# ============================================================
try:
    df = pd.read_csv(input_file, parse_dates=["Timestamp"], comment="#")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if df["Timestamp"].dt.tz is None:
        df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
    else:
        df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC")

except Exception as e:
    print_log("error", f"Failed to read CSV: {e}")
    sys.exit(1)

if "EventID" not in df.columns:
    print_log("error", "Input CSV must contain 'EventID' column.")
    sys.exit(1)

if args.tstart and args.tend:
    t_start = pd.Timestamp(args.tstart, tz="UTC")
    t_end = pd.Timestamp(args.tend, tz="UTC")
    print(f"\n[INFO] Filtering events between {t_start} and {t_end} (UTC)")
    df = df[(df["Timestamp"] >= t_start) & (df["Timestamp"] <= t_end)]
elif args.tstart or args.tend:
    print("[WARN] Please specify both --tstart and --tend to enable time filtering.")
else:
    print("[INFO] No time range specified — processing all events.")

if args.src is not None:
    src = args.src.lower()
    if src in ['b0329', 'B0329', 'b0329+54', 'B0329+54', 'PSR_B0329+54', 'PSR B0329+54']:
        src_name1 = 'B0329+54'
        src_name2 = 'PSR B0329+54'
    elif src in ['crab', 'CrabGRP', 'b0531+21', 'B0531+21', 'PSR_B0531+21', 'PSR B0531+21']:
        src_name1 = 'B0531+21'
        src_name2 = 'Crab (PSR B0531+21)'
    else:
        src_name1 = src
        src_name2 = src
    df = df[df["Name_src"].str.lower().isin([src_name1.lower(), src_name2.lower()])]

if df.empty:
    print("[WARN] No events found after filtering.")
    exit(0)

print(f"[INFO] {len(df)} events within the specified range will be processed.\n")

# ============================================================
# Main Loop
# ============================================================
log_entries = []
cache_seen = set()  # avoid duplicate links across shifts

for idx, row in df.iterrows():
    event_id = str(row["EventID"]).strip()
    try:
        event_dt = datetime.strptime(event_id, "%Y%m%d_%H%M%SZ")
    except ValueError:
        print_log("warn", f"Invalid EventID format: {event_id}, skipping.")
        continue

    print_log("info", f"Searching for event {event_id}")
    matched_files = []

    # --- Try exact match and shifted timestamps ---
    for shift_sec in range(0, args.shift_max + 1):
        shifted_dt = event_dt + timedelta(seconds=shift_sec)
        shifted_str = shifted_dt.strftime("%Y%m%d_%H%M%SZ")

        for search_dir in search_dirs:
            pattern = os.path.join(search_dir, f"*{shifted_str}*")
            matches = glob(pattern)
            for m in matches:
                if m not in cache_seen:
                    matched_files.append(m)
                    cache_seen.add(m)
            if args.debug and matches:
                print_log("debug", f"Matched {len(matches)} files for {shifted_str} in {search_dir}")

        if matched_files:
            break  # Stop once matches found for this event

    if not matched_files:
        print_log("warn", f"No matching files found for {event_id}")
        continue

    # --- Create output directory ---
    out_dir = os.path.join(".", event_id)
    if not args.dry_run:
        os.makedirs(out_dir, exist_ok=True)
    else:
        print_log("dryrun", f"Would create directory: {out_dir}")

    # --- Create symlinks or simulate ---
    for filepath in matched_files:
        filename = os.path.basename(filepath)
        link_path = os.path.join(out_dir, filename)

        if os.path.exists(link_path):
            print_log("Skip", f"Link already exists: {link_path}")
            continue

        if args.dry_run:
            print_log("dryrun", f"Would link: {filepath} → {link_path}")
        else:
            try:
                os.symlink(filepath, link_path)
                print_log("info", f"Linked: {filepath} → {link_path}")
            except Exception as e:
                print_log("error", f"Failed to link {filepath}: {e}")

    # --- Record summary log ---
    entry = {"EventID": event_id}
    for key in ["Timestamp", "Name_src", "DM", "BeamID"]:
        if key in df.columns:
            entry[key] = row[key]
    log_entries.append(entry)

# ============================================================
# Save Summary Log
# ============================================================
if args.dry_run:
    print_log("dryrun", f"Skipping log file write. Would save: {output_file}")
elif log_entries:
    if os.path.exists(output_file):
        print_log("warn", f"Overwriting existing log: {output_file}")
    pd.DataFrame(log_entries).to_csv(output_file, index=False)
    print_log("info", f"Saved log of {len(log_entries)} matched events → {output_file}")
else:
    print_log("warn", "No matches found. No log written.")
