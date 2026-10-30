"""Time windows to cut from the data (solar flares, ...).

The cut is made in gtmktime: every spacecraft row overlapping an excluded
window is dropped from the GTIs, so the photons AND the livetime of that
interval go together (counts map and livetime cube both read the filtered
GTI file). A cut applied later, on the binned maps, cannot do that.

Windows come from the 'time_exclusions' config section:

    time_exclusions:
      flsf_dir: /path/to/solar_flares_FLSF   # First Fermi-LAT Solar Flare catalog CSVs
      pad_hours: 1.0                          # added on both sides of every window
      windows_csv: data/input/sun_transients.csv   # UTC windows, one per row
      windows:                                # extra UTC windows, e.g. flares after 2018
        - ["2024-05-14T16:00", "2024-05-15T06:00"]

windows_csv needs 'start_utc' and 'stop_utc' columns (ISO UTC); any other
column is ignored.

FLSF (Ajello et al. 2021, ApJS 252, 13): 45 flares of 2010-2018, solar cycle
24 only. A flare's window runs from its GOES start to the end of its last
LAT detection window (Fermi sees the Sun ~40 min per orbit, so the emission
of a long flare spans several detection windows: the 2012-03-07 flare lasts
~20 h). LLE-only flares (prompt, minutes) get their trigger time.
"""
import csv
import hashlib
import os
from datetime import datetime, timedelta

# Fermi MET: seconds (TT) since MJDREF = 51910 + 7.428703703703703e-4 (TT),
# i.e. 2001-01-01 00:00:00 UTC.
MJDREFI, MJDREFF = 51910, 7.428703703703703e-4

FLSF_TIME_INTEGRATED = "TimeIntegratedResultsFLSF.csv"
FLSF_TIME_RESOLVED = "LikelihoodResultsFLSF.csv"
FLSF_LLE = "LLESEDFLSF.csv"
# Longest gap between consecutive LAT detection windows of one flare
# (catalogue max ~3.2 h: 2012-03-07, 2013-05-13b).
MAX_GAP = timedelta(hours=6)


def utc_to_met(dt):
    """datetime (UTC) or ISO string -> Fermi MET seconds (leap seconds included)."""
    from astropy.time import Time
    ref = Time(MJDREFI, MJDREFF, format='mjd', scale='tt')
    return float((Time(dt, scale='utc') - ref).sec)


def _read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def _hhmm_span(date, span):
    """'2011-03-07', '23:21 - 00:05' -> (start, stop) datetimes; stop rolls over midnight."""
    a, b = (s.strip() for s in span.split('-'))
    t0 = datetime.strptime(f"{date} {a}", "%Y-%m-%d %H:%M")
    t1 = datetime.strptime(f"{date} {b}", "%Y-%m-%d %H:%M")
    if t1 < t0:
        t1 += timedelta(days=1)
    return t0, t1


def flsf_windows(flsf_dir):
    """UTC (start, stop) per FLSF flare, unpadded and unmerged."""
    flares = []
    for row in _read_csv(os.path.join(flsf_dir, FLSF_TIME_INTEGRATED)):
        date = row['Main Name'].strip()[:10]
        flares.append(list(_hhmm_span(date, row['GOES Start-Stop'])))
    flares.sort()

    detections = []
    for row in _read_csv(os.path.join(flsf_dir, FLSF_TIME_RESOLVED)):
        date, span = row['Date and Time'].strip().split(' ', 1)
        detections.append(_hhmm_span(date, span))
    # Each LAT detection window belongs to the latest flare that started before
    # it, with 1 h slack (detection windows can open up to ~25 min before the
    # GOES start: 2011-08-09, 2012-11-13). It must also follow that flare's
    # previous window within MAX_GAP (one or two orbits), or a window that
    # misses its own flare would stretch an earlier one by days.
    loose = []
    for t0, t1 in sorted(detections):
        owner = [f for f in flares if f[0] <= t0 + timedelta(hours=1)]
        if owner and t0 <= owner[-1][1] + MAX_GAP:
            owner[-1][1] = max(owner[-1][1], t1)
        else:
            loose.append((t0, t1))

    # LLE-only flares: GBM trigger name bnYYMMDDfff, fff = fraction of the day.
    lle = []
    for row in _read_csv(os.path.join(flsf_dir, FLSF_LLE)):
        name = (row.get('') or next(iter(row.values()))).strip()
        if not name.startswith('bn'):
            continue
        day = datetime.strptime(name[2:8], "%y%m%d")
        t = day + timedelta(days=int(name[8:11]) / 1000.0)
        lle.append((t, t + timedelta(minutes=10)))

    return [tuple(f) for f in flares] + loose + lle


def csv_windows(path):
    """UTC (start, stop) per row of a CSV with 'start_utc'/'stop_utc' columns."""
    rows = _read_csv(path)
    if rows and not {'start_utc', 'stop_utc'} <= set(rows[0]):
        raise ValueError(f"time_exclusions.windows_csv {path}: needs 'start_utc' and "
                         f"'stop_utc' columns, has {sorted(rows[0])}")
    return [(datetime.fromisoformat(r['start_utc'].strip()),
             datetime.fromisoformat(r['stop_utc'].strip())) for r in rows]


def merge_windows(windows):
    """Sort and merge overlapping (start, stop) windows."""
    out = []
    for t0, t1 in sorted(windows):
        if out and t0 <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], t1))
        else:
            out.append((t0, t1))
    return out


def exclusion_windows(cfg):
    """MET (start, stop) windows to cut, padded and merged; [] if none configured."""
    cfg = cfg or {}
    pad = timedelta(hours=float(cfg.get('pad_hours', 0) or 0))
    utc = []
    if cfg.get('flsf_dir'):
        utc += flsf_windows(cfg['flsf_dir'])
    if cfg.get('windows_csv'):
        utc += csv_windows(cfg['windows_csv'])
    for w in cfg.get('windows') or []:
        if len(w) != 2:
            raise ValueError(f"time_exclusions.windows entry {w!r}: need [start, stop]")
        utc.append(tuple(datetime.fromisoformat(str(s)) for s in w))
    for t0, t1 in utc:
        if t1 <= t0:
            raise ValueError(f"time_exclusions: window {t0} - {t1} ends before it starts")
    padded = merge_windows([(t0 - pad, t1 + pad) for t0, t1 in utc])
    return [(utc_to_met(t0), utc_to_met(t1)) for t0, t1 in padded]


def gtmktime_exclusion(windows, tmin, tmax):
    """gtmktime filter suffix dropping every SC row that overlaps a window
    inside [tmin, tmax] (a week's TSTART/TSTOP): '' if none does.

    Only the week's own windows go in, so the expression stays a few terms
    long instead of carrying the whole list into every gtmktime call.
    """
    return "".join(f"&&!((START<{t1:.3f})&&(STOP>{t0:.3f}))"
                   for t0, t1 in windows if t0 < tmax and t1 > tmin)


def resolve_exclusion_tag(flsf_dir="", pad_hours=0, windows=None, windows_csv=""):
    """Product-name suffix: '' without exclusions, else e.g. '-xflsf1h',
    '-xflsf1h+2w3f9a1c' (2 extra windows, hash of their bounds) or
    '-xflsf1h+sun_transients64w5d0e2b' (64 windows from sun_transients.csv),
    so cut and uncut products never share a file name."""
    windows = list(windows or [])
    if not flsf_dir and not windows and not windows_csv:
        return ""
    tag = "-x" + ("flsf" if flsf_dir else "") + f"{float(pad_hours or 0):g}h"
    if windows:
        digest = hashlib.sha256(repr([[str(s) for s in w] for w in windows]).encode())
        tag += f"+{len(windows)}w{digest.hexdigest()[:6]}"
    if windows_csv:
        # Hash of the windows, not the file: editing other columns keeps the tag
        rows = csv_windows(windows_csv)
        digest = hashlib.sha256(repr([[str(s) for s in w] for w in rows]).encode())
        stem = os.path.splitext(os.path.basename(windows_csv))[0]
        tag += f"+{stem}{len(rows)}w{digest.hexdigest()[:6]}"
    return tag
