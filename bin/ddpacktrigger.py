#%matplotlib inline

# (Hiroto, 2025/10/21) ver1.0: Simplest ver. to directly read packed binary raw data and save it to a new file as a ddpacked data
# (Hiroto, 2025/10/21) ver1.1: [Debug] loadPackedBatchMmap(); shape=(nPack,) >> (bpp*nPack,)
# (Hiroto, 2025/10/22) ver1.2: [Revise] Add station for unique parameters 
# (Hiroto, 2025/10/30) ver1.3: [Revise] Improved version with --tstart and --tend for operational use

####  import necessary Modules ##########
import sys, os, os.path, time, re
import argparse
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec

import mmap
import numpy as np
from astropy.time import Time
from astropy.stats import sigma_clip
import pandas as pd

from packet_func import *

# (Hiroto, 2025/10/30) Functionalize getting station parameters
def get_station_params(station: str):
    """
    Return observatory/array configuration parameters for the specified station.

    Args:
        station (str): Station name (e.g., "Fushan", "Nantou")

    Returns:
        dict: Dictionary containing station parameters
    """

    # Common constants
    c = 2.998e8
    byteMeta = 64
    bytePack = 8256
    freq_ref = 400.0  # MHz
    BW = 400e6  # Hz

    # Default: Fushan
    params = {
        "nAnt": 16,
        "nRow": 16,
        "nChan": 1024,
        "nOrder": 8,
        "flim": [400., 800.],
        "frame_per_pack": 4,
        "byteMeta": byteMeta,
        "bytePack": bytePack,
        "freq_ref": freq_ref,
        "BW": BW,
    }

    if station.lower() == "Nantou":
        params.update({
            "nRow": 4,
            "nOrder": 2,
            "flim": [300., 700.],
            "frame_per_pack": 1,
        })

    # Derived quantities
    params["nSubCh"] = params["nChan"] // params["nOrder"]
    params["timeFrame"] = params["nChan"] / params["BW"]
    params["freq"] = np.linspace(params["flim"][0], params["flim"][1], params["nChan"], endpoint=False)
    params["lamb"] = c / (params["freq"] * 1e6)
    params["lamb0"] = c / (params["freq_ref"] * 1e6)
    params["PACKET_SIZE"] = params["bytePack"]
    params["meta"] = params["byteMeta"]

    return params

# (Hiroto, 2025/10/20) read_test: Revise to directly read packed binary raw data for ddpacktriggers.py
def loadPackedBatch(f, pack0, npack, bpp, meta=64):
    '''
    Read npack blocks of packed binary raw baseband data from an open file handle,
    starting from packet index pack0. Each block consists of a fixed-length
    header and payload, with a total size of bpp [bytes].

    input:
        f: open binary file handle (e.g., opened with 'rb')
        pack0: starting packet index (starting from 0)
        npack: number of packets (blocks) to read
        bpp: bytes per packet (header+payload; default = 8256)

    optional:
        meta: metadata length (in bytes) at the beginning of the file or buffer (default = 64)

    output:
        buf shape=(npack) : packed binary raw data (Header + packet) encoded in UInt8 with a total size of (header:64 + packet:8192) * npack [bytes]
    '''

    b0 = pack0 * bpp + meta
    f.seek(b0)
    buf = f.read(bpp*npack)
    return buf

def loadPackedBatchMmap(fname, nPack, bpp, skip_bytes):
    buf = np.memmap(fname, dtype='u1', mode='r', offset=skip_bytes, shape=(bpp*nPack,)) # UInt8
    return buf

# os.write ver.
def SaveBinaryFast(out_path, buf, overwrite=False):
    '''
    Efficiently and safely write binary data to a file.

    input:
        out_path: output file path
        buf: binary data (bytes, bytearray, or memoryview)
        overwrite: if True, overwrite existing file (default=False)

    behavior:
        - Uses low-level buffered write (os.open + os.write)
        - Automatically flushes and syncs data to disk
        - Prevents accidental overwrite unless specified
        - Supports large (>GB) binary data

    output:
        None
    '''
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise TypeError("buf must be bytes, bytearray, or memoryview")

    if os.path.exists(out_path) and not overwrite:
        os.remove(fout)

    # Open with low-level flags for performance
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    mode = 0o644  # -rw-r--r--
    fd = os.open(out_path, flags, mode)
    try:
        total_written = 0
        buf_mv = memoryview(buf)
        while total_written < len(buf):
            written = os.write(fd, buf_mv[total_written:])
            if written == 0:
                raise IOError("Disk write interrupted.")
            total_written += written
        os.fsync(fd)  # Ensure data is physically written to disk
    finally:
        os.close(fd)

# mmap ver.
def SaveBinaryFastMmap(out_path, buf, overwrite=False):
    '''
    Efficiently write binary data using memory-mapped I/O (mmap).

    input:
        out_path : output file path
        buf      : binary data (bytes, bytearray, or memoryview)
        overwrite: overwrite existing file if True (default=False)

    behavior:
        - Uses memory-mapped file for zero-copy write
        - Reduces CPU overhead by avoiding user–kernel copying
        - Suitable for large continuous binary data (e.g., >100MB)
        - Automatically syncs data and closes mapping safely

    output:
        None
    '''
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        raise TypeError("buf must be bytes, bytearray, or memoryview")

    if os.path.exists(out_path) and not overwrite:
        os.remove(fout)

    data_len = len(buf)
    # Step 1: ファイルを作成してサイズを確保
    with open(out_path, "wb") as f:
        f.truncate(data_len)

    # Step 2: 読み書きモードで開き直してmmapに対応
    with open(out_path, "r+b") as f:
        with mmap.mmap(f.fileno(), data_len, access=mmap.ACCESS_WRITE) as mm:
            mm.write(buf)
            mm.flush()

def get_sorted_bin_files(idir: str):
    '''
    Search for .bin files in the given directory and sort them
    by the numeric index after 'o' in their filenames (e.g., out_o0.bin).

    input:
        idir: input directory path

    output:
        files: sorted list of .bin file paths
    '''
    files = glob(f"{idir}/*.bin")
    files = sorted(
        files,
        key=lambda f: int(re.search(r'o(\d+)\.bin$', f).group(1))
        if re.search(r'o(\d+)\.bin$', f) else -1
    )
    return files


def make_ddpack_filenames(files):
    '''
    Generate output filenames by inserting "_ddpack" before ".bin".

    input:
        files: list of .bin file paths

    output:
        fouts: list of output file paths with "_ddpack" added
    '''
    fouts = []
    for f in files:
        base, ext = os.path.splitext(f)
        # fout = f"{base}_ddpack{ext}"
        fout = f"{f}.ddpack"
        fouts.append(fout)
    return fouts


def print_file_pairs(files, fouts):
    '''
    Print input and output file name pairs for verification.
    '''
    print(f"nFiles : {len(files)}")
    for fin, fout in zip(files, fouts):
        print(f"Input  : {fin}")
        print(f"Output : {fout}\n")

def pulse(DM, ep_event, freq, freq_ref):
    tau = 4148.8 * DM * (1/freq**2 - 1/freq_ref**2) # if freq>=freq_ref, then tau<0
    return ep_event+tau

# ============================================================
# Command line options
# ============================================================
parser = argparse.ArgumentParser(
    prog="ddpacktrigger.py",
    usage="%(prog)s csvfile [--tstart TSTART --tend TEND --station STATION --dry-run]",
    description="Convert packed binary raw data into ddpack format for event-based analysis.",
    epilog="Example:\n  python ddpacktrigger.py triggers.csv --tstart '2025-10-01 00:00:00' --tend '2025-10-01 01:00:00' --station Fushan",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("csvfile", help="CSV file containing event list")
parser.add_argument("--tstart", type=str, default=None,
                    help="Start time in format 'YYYY-MM-DD HH:MM:SS' (UTC)")
parser.add_argument("--tend", type=str, default=None,
                    help="End time in format 'YYYY-MM-DD HH:MM:SS' (UTC)")
parser.add_argument("--station", type=str, default="Fushan",
                    help="Specify station name to filter events (e.g., BURSTT11)")
parser.add_argument("--dry-run", action="store_true",
                    help="Show which events would be processed without executing any file operation")

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

# ============================================================
# Apply configuration based on station
# ============================================================

station = args.station or "Fushan"
params = get_station_params(station)
print(f"[INFO] Using station configuration: {station}")

# Unpack for readability
nAnt = params["nAnt"]
nRow = params["nRow"]
nChan = params["nChan"]
nOrder = params["nOrder"]
nSubCh = params["nSubCh"]
BW = params["BW"]
timeFrame = params["timeFrame"]
flim = params["flim"]
freq = params["freq"]
freq_ref = params["freq_ref"]
lamb = params["lamb"]
lamb0 = params["lamb0"]
byteMeta = params["byteMeta"]
bytePack = params["bytePack"]
frame_per_pack = params["frame_per_pack"]
PACKET_SIZE = params["PACKET_SIZE"]
meta = params["meta"]

# ============================================================
# Dedispersion / Windowing parameters
# ============================================================
# reading buffer before and after the pulse (raw data with dispersed pulse)
# the dedispersed pulse will be +/-secWin wide
secWin  = 0.03 #0.05 # in sec

pix_win = int(secWin * 2 / timeFrame)
pix_win = (pix_win // 4) * 4
print('pix_win:', pix_win)

# ============================================================
# Read CSV + Select specific date range
# ============================================================
dfile = args.csvfile
df = pd.read_csv(dfile, parse_dates=["Timestamp"], comment="#")


df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
if df["Timestamp"].dt.tz is None:
    df["Timestamp"] = df["Timestamp"].dt.tz_localize("UTC")
else:
    df["Timestamp"] = df["Timestamp"].dt.tz_convert("UTC")

if args.tstart and args.tend:
    t_start = pd.Timestamp(args.tstart, tz="UTC")
    t_end = pd.Timestamp(args.tend, tz="UTC")
    print(f"\n[INFO] Filtering events between {t_start} and {t_end} (UTC)")
    df = df[(df["Timestamp"] >= t_start) & (df["Timestamp"] <= t_end)]
elif args.tstart or args.tend:
    print("[WARN] Please specify both --tstart and --tend to enable time filtering.")
else:
    print("[INFO] No time range specified — processing all events.")

if df.empty:
    print("[WARN] No events found after filtering.")
    exit(0)

print(f"[INFO] {len(df)} events within the specified range will be processed.\n")

# ============================================================
# Dry-run MODE
# ============================================================
if args.dry_run:
    print("[DRY-RUN MODE] Listing events that would be processed:")
    print("-" * 90)
    for idx, row in df.iterrows():
        event_time = row["Timestamp"].strftime("%Y-%m-%d %H:%M:%S %Z")
        beamid_x = row["BeamID"] % 16
        beamid_y = row["BeamID"] // 16
        # station_name = row.get("Station", "N/A")
        station_name = station
        print(f"  EventID: {row['EventID']} | Time: {event_time} | "
              f"DM={row['DM']:.3f} | Beam=({beamid_x},{beamid_y}) | Station={station_name}")
    print("-" * 90)
    print(f"[INFO] Total {len(df)} events listed. No files were processed.\n")
    exit(0)


for idx, row in df.iterrows():
    ev_name = row['EventID']
    idir= ev_name
    dt_event = Time(row['Timestamp'], format='datetime', scale='utc')
    ep_event = dt_event.to_value('unix')
    DM = row['DM']
    beamid = row['BeamID']; beamid_x = beamid%16; beamid_y = beamid//16
    print(f"{row['EventID']} | {row['Timestamp']} | DM={row['DM']:.3f} | Beam=({beamid_x},{beamid_y})")


############
    files   = get_sorted_bin_files(idir)
    ddfiles = make_ddpack_filenames(files)
    print_file_pairs(files, ddfiles) # print out for check

    ###### Simulated Pulse ########
    ################################
    # Incoherent Dedispersion ver. # 
    ################################    
    ep_pulse = pulse(DM, ep_event, freq, freq_ref)
    plt.plot(ep_pulse, freq)
    plt.title(f"Expected Pulse of Event: {ev_name}")
    plt.savefig(f'Expected_dispersion_{ev_name}.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    nFile = len(files)
    for i in range(nFile):
        fbin = files[i]
        print(f"Input: {fbin}")
##################################        
        try:
            sz = os.path.getsize(fbin)   # follows symlink; raises if target missing
        except (OSError, FileNotFoundError) as e:
            print(f"Skipping broken/missing symlink: {fbin}  ({e})")
            continue
        
        # skip zero-length target files
        if sz == 0:
            print(f"Skipping (size=0): {fbin}")
            continue
############################
        packMax = (sz - byteMeta) // bytePack
    
        with open(fbin, 'rb') as f:
            mt = f.read(64)
            hd = f.read(64)
            tmp = decHeader2(hd)
            od = tmp[4]
            print(f'\n--- Processing File {i}: {fbin} (order {od}) ---')
    
            ep_begin = tmp[2] + 2 + (tmp[0] - tmp[4]) // nOrder * frame_per_pack * timeFrame
            ep_end = ep_begin + packMax // nRow * frame_per_pack * timeFrame
            ch_l = nSubCh * od
            ch_h = nSubCh * (od + 1) - 1
            ep_l = ep_pulse[ch_l]
            ep_h = ep_pulse[ch_h]
    
            # p1, p2 are nearest packet to the requested frame
            ## fix: increase margin to make sure [p1,p2] convers the window
            p1 = int((ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame) - 1) * nRow
            p2 = int((ep_l + secWin - ep_begin) / (frame_per_pack * timeFrame) + 1) * nRow
            nPack = p2 - p1
            nFrame = int(nPack / nRow * frame_per_pack)
            nPack = nFrame * int(nRow / frame_per_pack)
            p2 = p1 + nPack
    
            print(f'File {i}: packMax={packMax}, ep_begin={ep_begin:.4f}, ep_end={ep_end:.4f}')
            print(f'File {i}: ch_l={ch_l}, ch_h={ch_h}, ep_l={ep_l}, ep_h={ep_h}, secWin={secWin}, frame_per_pack={frame_per_pack}, timeFrame={timeFrame}, nRow={nRow}')
            print(f'File {i}: p1={p1}, p2={p2}, nPack={nPack}, nFrame={nFrame}')
    
            ## nFrame should be multiple of 16
            p0 = p1
            nPack = int(nFrame / 16) * 16 * 4
            skip_bytes = meta + PACKET_SIZE * p0

            # Safety: check file size
            sz = os.path.getsize(fbin)
            total_bytes = skip_bytes + PACKET_SIZE * nPack
            print(f"sz: {sz}")
            print(f"total_bytes: {total_bytes}")
            print(f"total_bytes > sz: {total_bytes > sz}")
            print(f"read_bytes: {PACKET_SIZE*nPack}")
            if total_bytes > sz:
                print(f"[SKIP] Requested mmap size ({total_bytes}) exceeds file size ({sz}) for {fbin}")
                continue

            # Load packed batch with safety
            try:
                # method1
                packs = loadPackedBatchMmap(fbin, nPack, PACKET_SIZE, skip_bytes)  # UInt8                
                # method2
                # packs = loadPackedBatch(f, p0, nPack, PACKET_SIZE, hdlen=64, meta=64) # UInt8
            except (ValueError, OSError) as e:
                print(f"[SKIP] Failed to mmap {fbin}: {e}")
                continue

            mt  = np.frombuffer(mt, dtype=np.uint8)
            buf = np.concatenate([mt, packs])

        fout = ddfiles[i]
        try:
            SaveBinaryFastMmap(fout, memoryview(buf), overwrite=False)
            print(f"Output: {fout}")
        except FileExistsError:
            print(f"[SKIP] Output file already exists: {fout}")
        except Exception as e:
            print(f"[ERROR] Failed to save {fout}: {e}")
            continue