#!/usr/bin/env python

# (Hiroto, 2025/10/30) ver2.0: [Revise] Improved version with --tstart and --tend for operational use
# (Hiroto, 2025/11/02) ver2.1: [Revise] Check whether the directories or files exist
# (Hiroto, 2025/11/03) ver2.2: [Debug] Temoral debug p1=int((ep_h-secWin-ep_begin)/(frame_per_pack*timeFrame)-1)*nRow >> p1=0
# (Hiroto, 2025/11/03) ver2.3: [Revise] Enforce lower limit: amp=np.clip(amp, 1e-12, None) | Add constrained_layout=True and delete plt.tight_layout()
# (Hiroto, 2025/11/06) ver2.4: [Revise] Add an --odir and --indir option for .npz and .ddpack | Revise .npz >> .ddpack.npz | idir= ev_name >> os.path.join(args.odir, ev_name)
# (Hiroto, 2026/01/20) ver2.6: [Debug] if (p2 > packMax):p2 = packMax

####  import necessary Modules ##########

import sys, os.path, time, re
import argparse
import subprocess
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import gridspec
from glob import glob
from packet_func import *
#print(sys.path)
#sys.path.append('/data/kylin/beamform')
# sys.path.append('/data/hmasaoka/analysis/ddpacktrigger/beamform')
sys.path.append('../beamform')
import beamform as bf
#help(bf)
from astropy.stats import sigma_clip

import pandas as pd
from astropy.time import Time

# ============================================================
# Logging Utility
# ============================================================
def print_log(level: str, msg: str):
    """Unified logging with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level.upper():7}] {msg}")

# ============================================================
# Functions
# ============================================================
def load_ddpack_files(idir):
    if not os.path.isdir(idir):
        print(f"[WARN] Directory not found: {idir}")
        return []

    # Get *.ddpack
    files = glob(os.path.join(idir, "*.ddpack"))
    if not files:
        print(f"[INFO] No .ddpack files found in {idir}")
        return []
    
    # Sort files based on o<number>.ddpack
    def extract_order(f):
        m = re.search(r'o(\d+)\.ddpack$', os.path.basename(f))
        return int(m.group(1)) if m else -1

    files = sorted(files, key=extract_order)
    nFile = len(files)

    print(f"[INFO] Found {nFile} .ddpack files in {idir}")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    return files

def parse_cal_time_astropy(fname):
    """
    cal_YYYYMMDD_HHMMSSZ.check → astropy Time (UTC)
    """
    base = os.path.basename(fname)
    tstr = base.replace("cal_", "").replace(".check", "")
    return Time.strptime(tstr, "%Y%m%d_%H%M%SZ", scale="utc")

#Difine observatory/Array design and parameters

nAnt = 16
nRow = 16
nChan = 1024
nOrder = 8
nSubCh = nChan//nOrder

BW = 400e6 # banwdith in Hz
timeFrame = nChan/BW # seconds per frame

flim = [400., 800.] # MHz
freq = np.linspace(flim[0], flim[1], nChan, endpoint=False)
freq_ref = 400. # bonsai trigger event time reference freq (MHz)

sepX = 1.0 # antenna separation in X, meters
sepY = 0.5 # in Y, meters
pos1 = np.arange(nAnt)*sepX
pos2 = np.arange(nRow)*sepY

lamb = 2.998e8 / (freq * 1e6) # wavelength in meters
lamb0 = 2.998e8/(freq_ref*1e6) # at lowest freq

byteMeta = 64
bytePack = 8256
frame_per_pack = 4 # bf256 case


PACKET_SIZE = bytePack
meta = byteMeta
N_CHANNEL = nSubCh
FL2INT = 256 # converting float parameter to integer matrix
SCALE = 1.0 / (FL2INT * FL2INT)

# reading buffer before and after the pulse (raw data with dispersed pulse)
# the dedispersed pulse will be +/-secWin wide
secWin = 0.03 # 0.020 # in sec

# time and channel binning
nSum = 400 # frame binning: 400-->1ms
nBin = 8   # chan binning

bfreq = freq.reshape(-1,nBin).mean(axis=1)
timeSamp = timeFrame * nSum # sec

cal_dir = "/burstt14/disk12/2nd_cal"
cal_files = glob(os.path.join(cal_dir, "cal_*.check"))

# ============================================================
# Command line options
# ============================================================
parser = argparse.ArgumentParser(
    prog="auto_beamform_FUSHAN_fix_ddpack.py",
    usage="%(prog)s csvfile [--tstart TSTART --tend TEND --station STATION]",
    description="Read ddpacked triggered baseband data and execute 2nd beamforming with a Chih-Yi's Fast Assenbly Code",
    epilog="Example:\n  python %(prog)s triggers.csv --tstart '2025-10-01T00:00:00' --tend '2025-10-01T01:00:00'",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("csvfile", help="CSV file containing event list")
parser.add_argument("--tstart", type=str, default=None,
                    help="Start time in format 'YYYY-MM-DDTHH:MM:SS' (UTC)")
parser.add_argument("--tend", type=str, default=None,
                    help="End time in format 'YYYY-MM-DDTHH:MM:SS' (UTC)")
parser.add_argument("--indir", type=str, default=".",
                    help="Input directory for .ddpack")
parser.add_argument("--odir", type=str, default=".",
                    help="Output directory for .npz")
# parser.add_argument("--station", type=str, default="Fushan",
#                     help="Specify station name to filter events (e.g., BURSTT11)")

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
# Load input CSV + Select specific date range
# ============================================================
try:
    dfile = args.csvfile
    df = pd.read_csv(dfile, parse_dates=["Timestamp"], comment="#")
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

if df.empty:
    print("[WARN] No events found after filtering.")
    exit(0)

print(f"[INFO] {len(df)} events within the specified range will be processed.\n")

# Print events

for idx, row in df.iterrows():
    ev_name = row['EventID']
    # idir= ev_name
    idir= os.path.join(args.indir, ev_name)
    dt_event = Time(row['Timestamp'], format='datetime', scale='utc')
    ep_event = dt_event.to_value('unix')
    DM = row['DM']
    beamid = row['BeamID']; beamid_x = beamid%16; beamid_y = beamid//16
    #beamid_x = row['beam_x']
    #beamid_y = row['beam_y']
    #print(f"{row['EventID']} | {row['Timestamp']} | DM={row['DM']:.3f} | Beam=({row['beam_x']},{row['beam_y']})")
    print(f"{row['EventID']} | {row['Timestamp']} | DM={row['DM']:.3f} | Beam=({beamid_x},{beamid_y})")
    #cal2_dir = '/home/sdutta/fushan/cal_20250709_023000Z.check'  ####### defining delay calibration #####
    #cal2_dir = '/home/sdutta/fushan/cal_20250820_033210Z.check'  ####### defining delay calibration #####
    #cal2_dir = '/home/sdutta/fushan/cal_20250901_031906Z.check'  ####### defining delay calibration #####
    #cal2_dir = '/home/sdutta/fushan/cal_20250917_032613Z.check'  ####### defining delay calibration #####
    # cal2_dir = 'cal_20250917_032613Z.check'  ####### defining delay calibration #####
    # cal2_dir = 'cal_20251017_053050Z.check'  ####### defining delay calibration #####

    # Automatically select a sutable calibration file
    candidates = []
    for f in cal_files:
        try:
            t_cal = parse_cal_time_astropy(f)
            if t_cal <= dt_event:
                candidates.append((t_cal, f))
        except ValueError:
            pass

    if not candidates:
        raise RuntimeError("No calibration files before this observation")

    # 最も直近（過去側）
    cal2_time, cal2_dir = max(candidates, key=lambda x: x[0].unix)

    print(f"Selected calibration: {cal2_dir}")
    print("Calibration time:", cal2_time.isot)
    print("Event time:", dt_event.isot)

    #[sdutta@burstt11 calibration]$/data/kylin/241212_new_bf256/calibration/2nd_cal_2509171126/cal_20250917_032613Z.check

#    if os.path.isdir(ev_name):
#        print(f"Processing {event_name} in directory {event_dir} ...")
#        os.chdir(ev_name)


##########33

    files = load_ddpack_files(idir)
    nFile = len(files)
    print(files)

    if nFile == 0:
        print("[WARN] No files to process — skipping this directory.")
        continue
    
    
    def get_2nd_cal(ifile, freq):
        '''
        load the delay calibration between FPGAs
        ifile lists the delay of each FPGA in ns
        freq is an array (length nChan) in MHz
    
        return
        '''
        nRow = 16
        nChan = len(freq)
    
        with open(ifile, 'r') as fh:
            tmp = fh.readline() # skip the first line
            line = fh.readline()
            tmp = line.split("'")[1].split()
            tau = np.array([float(x) for x in tmp])
            print('delays (ns):', tau)
    
        cal = np.exp(2.j*np.pi*tau.reshape(-1,1)*freq.reshape(1,-1)*1e-3)
    
        return cal
    
    cal2 = get_2nd_cal('%s/ant_delay_correct.txt'%cal2_dir, freq)
    
    ###### Simulated Pulse ########
    def pulse(DM, ep_event, freq, freq_ref):
        tau = 4148.8 * DM * (1/freq**2 - 1/freq_ref**2) # if freq>=freq_ref, then tau<0
        return ep_event+tau
    
    ep_pulse = pulse(DM, ep_event, freq, freq_ref)
    
    plt.plot(ep_pulse, freq)
    plt.title(f"Expected Pulse of Event: {ev_name}")
    plt.savefig(f'Expected_dispersion_{ev_name}.png', dpi=300, bbox_inches='tight')
    #plt.show()
    
    
    Xbeam0 = -7.5
    nXBeam = nAnt
    sin_theta_X = lamb0/sepX/nAnt*(np.arange(nXBeam)+Xbeam0)
    theta_X = np.arcsin(sin_theta_X)    # rad
    angles_X = theta_X / np.pi * 180.   # deg
    print('X angles (deg):', angles_X)
    
    ########  PLot dispersed pulse and dedispersed pulse #######
    ## the 2nd BFM
    # check 2nd beamform
    Ybeam0 = -7.5
    nYBeam = nRow
    
    # shape = (nYBeam,)
    sin_theta_Y = lamb0/sepY/nRow*(np.arange(nYBeam)+Ybeam0)
    theta_Y = np.arcsin(sin_theta_Y)    # rad
    angles_Y = theta_Y / np.pi * 180.   # deg
    print('Y angles (deg):', angles_Y)
    
    # shape = (nYBeam, nRow, nChan)
    BFMY = np.exp(2.j*np.pi*pos2.reshape(1,nRow,1)*sin_theta_Y.reshape(nYBeam,1,1)/lamb.reshape(1,1,nChan))
    BFMY = BFMY * cal2.reshape(1,nRow,nChan)
    ##########
    
    ################################
    #          Raw data ver.       # 
    ################################
    
    ## normalize rows
    print('reading fpga row amp:')
    amp = np.zeros((nRow, nChan))
    N_SAMPLE = 1024
    for fi in range(nFile):
        DATA_FILE = files[fi]
        ##################################
        try:
            sz = os.path.getsize(DATA_FILE)   # follows symlink; raises if target missing
        except (OSError, FileNotFoundError) as e:
            print(f"Skipping broken/missing symlink: {DATA_FILE}  ({e})")
            continue
        # skip zero-length target files
        if sz == 0:
            print(f"Skipping (size=0): {DATA_FILE}")
            continue
        ############################
        with open(DATA_FILE, 'rb') as fh:
            mt = fh.read(64)
            hd = fh.read(64)
            tmp = decHeader2(hd)
            od = tmp[4]
        ch1 = nSubCh*od
        ch2 = ch1 + nSubCh
        print('... order:',od)
    
        fpga = np.memmap(DATA_FILE, dtype='u1', mode='r')
        p0 = 0
        data_off = meta + PACKET_SIZE * p0
    
        v1 = np.zeros((N_CHANNEL, nAnt, 2), dtype='<i2')
        v1[:, beamid_x, 0] = FL2INT
    
        for ri in range(nRow):
            data = fpga[data_off:]
            v2 = np.zeros((N_CHANNEL, nRow, 2), dtype='<i2')
            v2[:, ri, 0] = FL2INT
    
            res = np.ones((N_CHANNEL, N_SAMPLE), dtype='<c8')
            bf.beamform256(res, data, v1, v2, N_SAMPLE, SCALE)  # read baseband data & 2nd beamforming
            amp[ri,ch1:ch2] = np.abs(res).mean(axis=1)          # Integration in a time-domain

            # print(f"res: {res.shape}, {res.dtype}")
            # print(f"data: {data.shape}, {data.dtype}")
            # print(f"v1: {v1.shape}, {v1.dtype}")
            # print(f"v2: {v2.shape}, {v2.dtype}")
            # print(f"N_SAMPLE: {N_SAMPLE}")
            # print(f"SCALE: {SCALE}")

    
    ## renormalize amp so that median of rows is 1
    med_amp = np.median(amp, axis=1)    # med of each row
    med_med_amp = np.median(med_amp)    # med of all rows
    amp /= med_med_amp
    
    ## normalize BFMY according to the amp
    ## and zero out channels or rows that have low sensitivity
    amp = np.clip(amp, 1e-12, None)  # (Hiroto, 2025/11/03) Enforce lower limit 
    BFMY = BFMY / amp.reshape(1,nRow,nChan)
    w_low_amp = amp<0.2
    for i in range(nYBeam):
        BFMY[i][w_low_amp] *= 0.
    
    fig, s2d = plt.subplots(4,4,figsize=(10,6),sharex=True,sharey=True)
    sub = s2d.flatten()
    for ri in range(nRow):
        ax = sub[ri]
        ax.plot(freq, amp[ri])
        ax.text(0.05, 0.85, 'row%02d'%(ri+1), transform=ax.transAxes)
        if (ri%4==0):
            ax.set_ylabel('amp')
        if (ri>11):
            ax.set_xlabel('freq (MHz)')
    fig.suptitle('beamid_x=%d'%beamid_x)
    fig.tight_layout(rect=[0,0.03,1,0.95])
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(f'fpga_row_amp{ev_name}_{beamid_x}_{beamid_y}.png')
    plt.close(fig)
    print('... done.')
    
    ################################
    # Incoherent Dedispersion ver. # 
    ################################
    
    ## 1st beamform, identity matrix, select beamid_x
    ## 2nd beamform, pos+cal2, select beamid_y
    
    pix_win = int(secWin * 2 / timeFrame)
    pix_win = (pix_win // 4) * 4
    print('pix_win:', pix_win)
    
    dd_spec = np.zeros((nChan, pix_win), dtype=np.complex64)
    print('dd_spec.shape', dd_spec.shape)
    
    nSubBin = nSubCh//nBin
    
    combined_dd_inten = []
    combined_freq = []      # To store frequencies per file
    win_len_list = []       # Keep track of dedispersion window lengths per file
    
    
    for i in range(nFile):
        fbin = files[i]
##################################        
        #sz = os.path.getsize(fbin)
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
    
        with open(fbin, 'rb') as fh:
            mt = fh.read(64)
            hd = fh.read(64)
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
            """
            (Hiroto, 2025/11/03) 
            p1 should be zero because only pulse data is retrived by ddpacktrigger.py.
            However, p1=16 sometimes show up for some reason, which causes a process crush.
            Furthermore, I found a p1 value of ddpack became the same as that of bin for some reason.
            I temoporaly solve these issues by replacing if p1 == 16: p1=0.
            These issues should be understood well and solved properly.
            """
            p1 = int((ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame) - 1) * nRow # (Hiroto, 2025/11/03) sometimes p1=16 >>Error
            if p1 == 16: # (Hiroto, 2025/11/03) sometimes p1=16, which causes a process crush
                p1 = 0   # (Hiroto, 2025/11/03) p1=0 >> No error
            p2 = int((ep_l + secWin - ep_begin) / (frame_per_pack * timeFrame) + 1) * nRow
            nPack = p2 - p1
            nFrame = int(nPack / nRow * frame_per_pack)
            nPack = nFrame * int(nRow / frame_per_pack)
            p2 = p1 + nPack

            if (p2 > packMax):
                p2 = packMax
    
            print(f'File {i}: packMax={packMax}, ep_begin={ep_begin:.4f}, ep_end={ep_end:.4f}')
            print(f'File {i}: ch_l={ch_l}, ch_h={ch_h}, ep_l={ep_l}, ep_h={ep_h}, secWin={secWin}, frame_per_pack={frame_per_pack}, timeFrame={timeFrame}, nRow={nRow}')
            print(f'File {i}: p1={p1}, p2={p2}, nPack={nPack}, nFrame={nFrame}')
            
            test = (ep_h - secWin - ep_begin) / (frame_per_pack * timeFrame) - 1
            print(f'File {i}: p1_row={test}')

            ## nFrame should be multiple of 16
            nPack = int(nFrame / 16) * 16 * 4
            nFrame = nPack / 4
            N_SAMPLE = int(nFrame)
            DATA_FILE = fbin
            fpga = np.memmap(DATA_FILE, dtype='u1', mode='r')
            p0 = p1
            data_off = meta + PACKET_SIZE * p0
            data = fpga[data_off:]
            v1 = np.zeros((N_CHANNEL, nAnt, 2), dtype='<i2')
            v1[:, beamid_x, 0] = FL2INT
            BFMYvec = BFMY[beamid_y, :, ch_l:ch_l + nSubCh].T * FL2INT
            v2 = BFMYvec.ravel().view('<f8').reshape((nSubCh, nRow, 2)).astype('<i2')
            res = np.ones((N_CHANNEL, N_SAMPLE), dtype='<c8')
            bf.beamform256(res, data, v1, v2, N_SAMPLE, SCALE)
            nTime1 = res.shape[1]//nSum
            res_shape1 = nTime1 * nSum
            ## dispersed waterfall
            inten = np.abs(res[:, :res_shape1]).reshape(nSubCh, nTime1, nSum).mean(axis=2)
            norm_inten = inten / np.mean(inten, axis=1, keepdims=True) #Normalize only
            #########
            # Dedispersion step (using channel-specific shifts)
            # for baseband
            #pix_start = np.array((ep_pulse[ch_l:(ch_l+nSubCh)] - ep_h) / timeFrame).astype(int)
            ## original pix_start will always be 0 at highest freq channel
            ## fix: the reference time is not ep_h (epoch time of the extracted data at the highest freq)
            ## instead, it should be computed from p1 + secWin
            ## with the added margin on p1, the pix_start here should be non-negative
            ep1 = ep_begin + (p1//nRow)*frame_per_pack*timeFrame
            pix_start = np.array((ep_pulse[ch_l:(ch_l+nSubCh)] - secWin - ep1) / timeFrame).astype(int)
            print('first pix:', pix_start[-1])
    
            for chi in range(nSubCh):
                ch = chi + ch_l
                pp = pix_start[chi]
                tmp = res[chi, pp:pp+pix_win]
                max_len = tmp.size
                dd_spec[ch,:max_len] = tmp[:]
    
    
            ## dedispersed waterfall
            nTime = pix_win//nSum
            dd_len = nTime*nSum
            #print('nTime, dd_len', nTime, dd_len)
            dd_inten = np.abs(dd_spec[ch_l:ch_l+nSubCh,:dd_len]).reshape(nSubBin, nBin, nTime, nSum).mean(axis=(1,3))
            dd_inten /= dd_inten.mean(axis=1, keepdims=True)
            #print('dd_inten.shape', dd_inten.shape)
            #print('bfreq', bfreq[ch_l//nBin:(ch_l+nSubCh)//nBin])
    
            #print(dd_inten)
            combined_dd_inten.append(dd_inten)
            combined_freq.append(bfreq[ch_l//nBin:(ch_l+nSubCh)//nBin])
    
            # Compute time axis for raw data plot
            time_per_bin_ms = nSum * timeFrame * 1000  # ms
            print(f'time_per_bin_ms={time_per_bin_ms}')
            raw_time_axis = np.arange(inten.shape[1]) * time_per_bin_ms  # ms
            dedisp_time_axis = np.arange(dd_inten.shape[1]) * time_per_bin_ms  # ms
    
            #fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
            # 1. Dispersed Pulse (before dedispersion)
            #pcm1 = axs[0].pcolormesh(raw_time_axis, freq[ch_l:ch_l + nSubCh], clipped_inten, shading='auto') #normalized plot
            pcm1 = axs[0].pcolormesh(raw_time_axis, freq[ch_l:(ch_l+nSubCh)], norm_inten, shading='auto') #normalized plot
            #pcm1 = axs[0].pcolormesh(raw_time_axis, freq[ch_l:ch_l + nSubCh], inten, shading='auto') #NON normalized plot
            axs[0].set_title(f'Dispersed Pulse (File {i}, Order {od})')
            axs[0].set_xlabel('Time (ms)')
            axs[0].set_ylabel('Frequency (MHz)')
            fig.colorbar(pcm1, ax=axs[0], label='Normalized Amplitude')
    
            # 2. Dedispersed Pulse (after dedispersion)
            pcm2 = axs[1].pcolormesh(dedisp_time_axis, bfreq[ch_l//nBin:(ch_l+nSubCh)//nBin], dd_inten, shading='auto')
            #print(dd_inten)
            #pcm2 = axs[1].pcolormesh(dd_inten, shading='auto')
            #pcm2 = axs[1].pcolormesh(raw_time_axis, freq[ch_l:(ch_l+nSubCh)], norm_inten, shading='auto') #normalized plot
            axs[1].set_title(f'Dedispersed Pulse (File {i}, Order {od})')
            axs[1].set_xlabel('Time (ms)')
            axs[1].set_ylabel('Frequency (MHz)')
            fig.colorbar(pcm2, ax=axs[1], label='Normalized Amplitude')
            fig.suptitle(f"Dedispersed Pulse of Event: {ev_name}-file{i}")
            #fig.title(f'Dedispersed Pulse of Event: {ev_name}-file{i}')
            plt.savefig(f'pulse_dedispersion_single_{ev_name}_file{i}.png', dpi=300, bbox_inches='tight')
    
    
    
    # === save dedispersed baseband ===
    ep0 = ep_event - secWin # begin epoch of the dedispersed spec, at 400MHz
    dd_time = np.arange(pix_win)*timeFrame # time offset of each frame
    # out_name = 'baseband_beamform_dedisperse_%s.ddpack.npz'%(ev_name,)
    out_name = '%s/baseband_beamform_dedisperse_%s.ddpack.npz'%(args.odir, ev_name)
    np.savez(out_name, freq=freq, ep0=ep0, dd_time=dd_time, dd_spec=dd_spec, ep_event=ep_event, DM=DM, beamid_x=beamid_x, beamid_y=beamid_y)
    
    
    # === Data Processing ===
    combined_dd_inten_stack = np.vstack(combined_dd_inten)
    std = np.std(combined_dd_inten_stack)
    mean = np.mean(combined_dd_inten_stack)
    norm_combined_dd_inten_stack = (combined_dd_inten_stack - mean) / std
    clipped_inten = np.clip(norm_combined_dd_inten_stack, -4, 4)
    
    combined_freq = np.concatenate(combined_freq)
    
    # Axis edges
    freq_step = np.diff(combined_freq).mean()
    freq_edges = np.hstack([combined_freq[0] - freq_step/2, combined_freq + freq_step/2])
    time_axis = np.arange(combined_dd_inten_stack.shape[1]) * time_per_bin_ms
    time_edges = np.hstack([time_axis[0] - time_per_bin_ms/2, time_axis + time_per_bin_ms/2])
    
    # === SNR Calculation (positive) ===
    # signal = np.max(clipped_inten, axis=0)
    # noise = np.std(clipped_inten, axis=0)
    #signal = np.max((combined_dd_inten_stack-mean), axis=0)
    #noise = np.std((combined_dd_inten_stack-mean), axis=0)
    ####
    # signal = np.sum((norm_combined_dd_inten_stack), axis=0)
    # #noise = np.std((norm_combined_dd_inten_stack), axis=0)
    # noise = np.std(norm_combined_dd_inten_stack)  # single scalar noise level
    # snr_time = signal / noise
    # snr_time -= np.median(snr_time)
    #

    # clipped = np.clip(norm_combined_dd_inten_stack[0:256,:], 0, 10) # 400-500MHz (nBin=1)
    # clipped = np.clip(norm_combined_dd_inten_stack[0:256,:], 0, 100) # 400-500MHz (nBin=1), Strong CrabGRPs; SNR>100
    clipped = np.clip(norm_combined_dd_inten_stack, 0, 10) # 400-800MHz
    # clipped = np.clip(norm_combined_dd_inten_stack, 0, 100) # 400-800MHz, Strong CrabGRPs; SNR>100

    signal = np.mean((clipped), axis=0)
    #noise = np.std((norm_combined_dd_inten_stack), axis=0)
    #noise = np.std(clipped)  # single scalar noise level
    noise = sigma_clip(signal).std()
    snr_time = signal / noise
    snr_time -= np.median(snr_time)
    #####
    # # Center each time bin by subtracting mean across frequency:
    # centered = norm_combined_dd_inten_stack - norm_combined_dd_inten_stack.mean(axis=0, keepdims=True)
    # signal = np.max(centered, axis=0)
    # noise = np.std(centered, axis=0)
    # snr_time = signal / noise
    #####
    # # Centered data (already normalized)
    # centered = norm_combined_dd_inten_stack - norm_combined_dd_inten_stack.mean(axis=0, keepdims=True)
    # # Signal is now the *sum* across frequency (broadband enhancement)
    # signal = np.sum(centered, axis=0)
    # # Noise is the std of the centered data across frequency
    # noise = np.std(centered, axis=0)
    # # SNR over time
    # snr_time = signal / noise
    # # Subtract median to zero the baseline
    # snr_time -= np.median(snr_time)
    # # from scipy.ndimage import gaussian_filter1d
    # # snr_time = gaussian_filter1d(snr_time, sigma=2)
    ######

    print("snr_time min/max:",
        np.nanmin(snr_time),
        np.nanmax(snr_time))    
    
    # === Plotting with GridSpec (3 columns: [plot, plot, colorbar]) ===
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[20, 1, 0.5], height_ratios=[4, 1], hspace=0.05, wspace=0.05)
    
    # --- Top Plot (pcolormesh) ---
    ax0 = fig.add_subplot(gs[0, 0])  # row 0, col 0
    pc = ax0.pcolormesh(time_edges, freq_edges, clipped_inten, cmap='plasma', shading='auto')
    ax0.set_title(f'Combined Dedispersed Pulse of Event: {ev_name}')
    ax0.set_ylabel('Frequency (MHz)')
    ax0.tick_params(labelbottom=False)
    
    # --- Colorbar aligned with ax0 only ---
    cax = fig.add_subplot(gs[0, 1])  # row 0, col 1
    cb = fig.colorbar(pc, cax=cax)
    cb.set_label('Normalized Amplitude')
    
    # --- Bottom Plot (SNR) aligned to ax0 width only ---
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)  # row 1, col 0
    ax1.plot(time_axis, snr_time, color='black', linewidth=1)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('SNR')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Finalize
    # plt.tight_layout()
    plt.savefig(f'pulse_dedispersion_combined_{ev_name}_ddpack.png', dpi=300)
    plt.close(fig)
    #plt.show()



