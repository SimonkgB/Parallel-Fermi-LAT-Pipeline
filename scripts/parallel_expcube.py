#!/usr/bin/env python3
"""Energy-parallel gtexpcube2: one process per energy bin, stacked at the end.

gtexpcube2 is single-threaded, but its exposure planes are independent per
energy bin - on a many-core machine the monolithic run is a pure serial tail.
This wrapper accepts the exact same key=value parameter list as gtexpcube2
(so a pipeline config only swaps the executable), runs one gtexpcube2 per
LOG energy bin in parallel, and stacks the per-bin outputs into a file with
the same structure as a monolithic run (image cube or HEALPix table, plus
ENERGIES and GTI extensions).

Usage (drop-in for a gtexpcube2 command line):
    python3 scripts/parallel_expcube.py [--cores N] \
        infile=lt.fits cmap=counts.fits outfile=exposure.fits \
        irfs=... evtype=3 bincalc=CENTER emin=100 emax=316228 enumbins=28 \
        <grid/hpx params forwarded verbatim>

Notes:
  - cmap is passed through to the per-bin runs: gtexpcube2 takes the map
    GEOMETRY (WCS grid or HEALPix) and DSS keywords from it, while the
    explicit emin/emax/enumbins parameters override the ENERGY binning -
    which is exactly what the per-bin runs exploit. Without a cmap the
    spatial grid must be given explicitly (nxpix/binsz/...).
  - Requires LOG energy binning (gtexpcube2's emin/emax/enumbins with
    ebinfile=NONE).
"""
import os
import sys
import time
import argparse
import subprocess
import tempfile
import shutil
import multiprocessing

import numpy as np
from astropy.io import fits

# make_worker_env lives with the pipeline runner (private PFILES per worker,
# without which parallel gtool startups contend on .par file locks);
# setup_fermi_environment makes standalone runs work outside the pipeline.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runners"))
from parallel_run import (make_worker_env, setup_fermi_environment,  # noqa: E402
                          seed_system_pfiles, Logger, APE_TRANSIENT_MARKER)


def run_one_bin(args):
    """Run gtexpcube2 for a single energy bin. Returns (idx, None) or (idx, error)."""
    idx, params, out_path = args
    cmd = ["gtexpcube2"] + [f"{k}={v}" for k, v in params.items()] + [f"outfile={out_path}"]
    env = make_worker_env()
    for attempt in (1, 2, 3):
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, env=env)
        if proc.returncode == 0:
            return (idx, None)
        if APE_TRANSIENT_MARKER not in (proc.stderr or "") or attempt == 3:
            break
        time.sleep(2 * attempt)
    return (idx, f"bin {idx}: {' '.join(cmd)}\nSTDERR: {proc.stderr.strip()}")


def stack_planes(plane_files, outfile):
    """Combine per-bin gtexpcube2 outputs into one monolithic-format file."""
    energies = []
    for f in plane_files:
        with fits.open(f) as hdul:
            energies.extend(np.asarray(hdul["ENERGIES"].data["Energy"], dtype=np.float64))

    with fits.open(plane_files[0]) as tpl:
        is_image = tpl[0].data is not None

        if is_image:
            data = np.concatenate([fits.getdata(f, 0) for f in plane_files], axis=0)
            primary = fits.PrimaryHDU(data=data, header=tpl[0].header.copy())
            # A 1-bin run writes CDELT3=0 (no axis step to infer); the stacked
            # log-energy axis step is the log bin width.
            if len(energies) > 1 and "CTYPE3" in primary.header:
                primary.header["CDELT3"] = float(np.log(energies[1] / energies[0]))
            extras = [h.copy() for h in tpl[1:] if h.name != "ENERGIES"]
        else:
            # HEALPix: one table column per bin (ENERGY1..ENERGYn)
            cols = []
            for i, f in enumerate(plane_files):
                with fits.open(f) as hdul:
                    tab = hdul[1]
                    src = tab.columns[0]
                    cols.append(fits.Column(name=f"ENERGY{i+1}", format=src.format,
                                            array=np.asarray(tab.data[src.name])))
            table = fits.BinTableHDU.from_columns(cols, header=tpl[1].header.copy(),
                                                  name=tpl[1].name)
            primary = fits.PrimaryHDU(header=tpl[0].header.copy())
            extras = [table] + [h.copy() for h in tpl[2:] if h.name != "ENERGIES"]

        with fits.open(plane_files[0]) as ref:
            energies_hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name="Energy", format="1D", array=np.array(energies))],
                header=ref["ENERGIES"].header.copy(), name="ENERGIES")

        # ENERGIES sits right after the map data, GTI (and anything else) after it
        gti_like = [h for h in extras if h.name == "GTI"]
        others = [h for h in extras if h.name != "GTI"]
        fits.HDUList([primary] + others + [energies_hdu] + gti_like).writeto(
            outfile, overwrite=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cores", type=int, default=0,
                    help="parallel gtexpcube2 processes (default: min(nbins, cpu_count))")
    ap.add_argument("--fermi-base", default=None,
                    help="fermitools env root (only needed standalone, outside the pipeline)")
    ap.add_argument("params", nargs="+", help="gtexpcube2 key=value parameters")
    args = ap.parse_args()

    # No-op when the pipeline already exported CALDB & co.; fixes them up for
    # standalone use (derived from gtexpcube2's location if --fermi-base unset).
    quiet = Logger(None)
    setup_fermi_environment(quiet, args.fermi_base)
    # Serve .par files from a local mirror: N parallel gtexpcube2 startups all
    # reading syspfiles over a network mount can fail with ape errors.
    seed_system_pfiles(quiet)

    params = {}
    for p in args.params:
        if "=" not in p:
            sys.exit(f"ERROR: expected key=value gtexpcube2 parameter, got: {p}")
        k, v = p.split("=", 1)
        params[k] = v

    for req in ("infile", "outfile", "emin", "emax", "enumbins"):
        if req not in params:
            sys.exit(f"ERROR: missing required parameter {req}=")

    outfile = params.pop("outfile")
    emin, emax = float(params.pop("emin")), float(params.pop("emax"))
    nbins = int(params.pop("enumbins"))
    # An unset cmap would make gtexpcube2 prompt and hang the pool
    params.setdefault("cmap", "none")

    edges = np.logspace(np.log10(emin), np.log10(emax), nbins + 1)
    cores = args.cores or multiprocessing.cpu_count()
    cores = max(1, min(cores, nbins))

    tmp_dir = tempfile.mkdtemp(prefix="parallel_expcube_")
    try:
        tasks = []
        for i in range(nbins):
            bin_params = dict(params)
            bin_params["emin"] = f"{edges[i]:.15g}"
            bin_params["emax"] = f"{edges[i+1]:.15g}"
            bin_params["enumbins"] = "1"
            tasks.append((i, bin_params, os.path.join(tmp_dir, f"plane_{i:03d}.fits")))

        print(f"parallel_expcube: {nbins} bins on {cores} cores...")
        with multiprocessing.Pool(cores) as pool:
            outcomes = pool.map(run_one_bin, tasks)

        errors = [err for _, err in outcomes if err]
        if errors:
            for err in errors:
                print(f"ERROR: {err}", file=sys.stderr)
            sys.exit(1)

        stack_planes([t[2] for t in tasks], outfile)
        print(f"parallel_expcube: wrote {outfile}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
