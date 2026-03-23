#!/usr/bin/env python

# (Hiroto, 2025/10/30) ver1.0: Unified automation script to sequentially execute the event processing pipeline.
# (Hiroto, 2025/10/31) ver1.1: Add --skip-ddpack option that allows you to analyze triggered data without ddpack
# (Hiroto, 2025/11/02) ver1.2: Update auto_beamform_FUSHAN_fix*ver2.0.py >> ver2.1
# (Hiroto, 2025/11/03) ver1.3: Update auto_beamform_FUSHAN_fix_ddpack*ver2.1.py >> ver2.3
# (Hiroto, 2025/11/06) ver1.4: Add --odir_ddpack and odir_npz options| Update ddpacktrigger_ver1.3.py >> ver1.4, auto_beamform_FUSHAN_fix_ver2.1.py >> ver2.2 |.  auto_beamform_FUSHAN_fix_ddpack_ver2.3.py >> ver2.4
# (Hiroto, 2025/11/28) ver1.5: [Revise] Update ddpack*ver1.4.py >> ver1.5
# (Hiroto, 2025/12/11) ver1.6: [Revise] Update link*ver2.0.py >> ver2.1 | Add --src option to specify a source

import subprocess
import argparse
import sys
from datetime import datetime

# ============================================================
# Command-line options
# ============================================================
parser = argparse.ArgumentParser(
    prog="auto_pipeline.py",
    usage="%(prog)s csvfile FILE --tstart TSTART --tend TEND [--odir_ddpack] [--odir_npz] [--station STATION] [--src TARGET] [--dry-run]",
    description=(
        "Run the full event processing pipeline in sequence:\n"
        "1. link_event_filesFUSHAN_ver2.1.py\n"
        "2. ddpacktrigger_ver1.5.py\n"
        "3. auto_beamform_FUSHAN_fix_ver2.2.py\n"
        "4. auto_beamform_FUSHAN_fix_ddpack_ver2.4.py\n"
    ),
    epilog=("Example:\n"
            "  python auto_pipeline.py triggers.csv "
            "--tstart '2025-10-16 00:00:00' --tend '2025-10-29 00:00:00' --station Fushan --src 'b0329'"),
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("csvfile", help="CSV file containing event list")
parser.add_argument("--tstart", required=True, help="Start time in UTC (format: 'YYYY-MM-DD HH:MM:SS')")
parser.add_argument("--tend", required=True, help="End time in UTC (format: 'YYYY-MM-DD HH:MM:SS')")
parser.add_argument("--odir_ddpack", type=str, default=".", help="Output directory for .ddpack")
parser.add_argument("--odir_npz", type=str, default=".", help="Output directory for .npz")
parser.add_argument("--station", default="Fushan", help="Station name (default: Fushan)")
parser.add_argument("--src", type=str, default=None, help="Select a specific astronomical source")
parser.add_argument("--skip-ddpack", action="store_true", help="Skip ddpack process")
parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")

# Show usage if no options
if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

# Clarify the csvfile position
if not args.csvfile:
    print("[ERROR] Missing required argument: csvfile\n")
    parser.print_usage(sys.stderr)
    sys.exit(1)

selected_csvfile = args.csvfile.replace(".csv", "_linked_files.csv")
# ============================================================
# Helper: execute command with logging
# ============================================================
def run_command(cmd_list, dry_run=False):
    print("\n" + "="*70)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Running command:")
    print(" ".join(cmd_list))
    print("="*70)
    
    if dry_run:
        print("[Dry-Run] Command not executed.")
        return 0

    try:
        result = subprocess.run(cmd_list, check=True, text=True)
        print(f"[INFO] Command finished successfully: {cmd_list[0]}")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {cmd_list[0]}")
        print(f"        Return code: {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"[FATAL] Unexpected error while running {cmd_list[0]}: {e}")
        sys.exit(1)

# ============================================================
# Define pipeline steps
# ============================================================
pipeline = [
    [
        "python", "link_event_filesFUSHAN_ver2.1.py",
        args.csvfile, "--tstart", args.tstart, "--tend", args.tend
    ]+ (["--src", args.src] if args.src is not None else []),
    [
        "python", "ddpacktrigger_ver1.5.py",
        selected_csvfile, "--tstart", args.tstart, "--tend", args.tend,
        "--odir", args.odir_ddpack, "--station", args.station
    ],
    [
        "python", "auto_beamform_FUSHAN_fix_ver2.2.py",
        selected_csvfile, "--tstart", args.tstart, "--tend", args.tend,
        "--odir", args.odir_npz
    ],
    [
        "python", "auto_beamform_FUSHAN_fix_ddpack_ver2.4.py",
        selected_csvfile, "--tstart", args.tstart, "--tend", args.tend,
        "--indir", args.odir_ddpack, "--odir", args.odir_npz
    ]
]

# ============================================================
# Execute pipeline
# ============================================================
print(f"\n[Pipeline Start] Processing {args.csvfile} ({args.station})")
for step_id, cmd in enumerate(pipeline, start=1):
    print(f"\n--- Step {step_id}/{len(pipeline)} ---")
    if step_id == 2 and args.skip_ddpack:
        print("[Skip] ddpack step skipped (--skip-ddpack specified).")
        continue
    run_command(cmd, dry_run=args.dry_run)

print("\n✅ [Pipeline Completed] All steps finished successfully.")