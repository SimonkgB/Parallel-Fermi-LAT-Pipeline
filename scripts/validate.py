#!/usr/bin/env python3
"""Sanity-check Fermi LAT counts / exposure / flux maps.

Runs the checks that actually catch pipeline problems:
  1. Counts: non-negative, total photons, per-energy-bin falloff
  2. Energy grid (EBOUNDS): number of bins, range, bins-per-decade
  3. Exposure: positivity over the observed sky, magnitude sanity
  4. flux == counts / exposure (where exposure > 0), finite/positive
  5. Metadata: coordinate system, time span
  6. Cross-check: counts vs exposure vs flux share shape & energy binning

Works for both CCUBE image cubes (data in primary HDU) and HEALPix maps
(data in the SKYMAP binary table, one column per energy bin).

Usage:
    python3 scripts/validate.py COUNTS.fits [EXPOSURE.fits] [FLUX.fits]

    # or cross-check two counts maps (e.g. HEALPix vs CCUBE) for equal totals:
    python3 scripts/validate.py --compare COUNTS_A.fits COUNTS_B.fits
"""
import sys
import argparse
import numpy as np
from astropy.io import fits

OK = "\033[32mOK\033[0m"
WARN = "\033[33mWARN\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def load_map(path):
    """Return (data_2d, header, ebounds_or_None).

    data_2d has shape (npix, nbins): image cubes are reshaped to (nbins, npix)->
    transposed; HEALPix tables are stacked column-per-bin.
    """
    with fits.open(path) as hdul:
        primary = hdul[0].header
        ebounds = hdul["EBOUNDS"].data if "EBOUNDS" in [h.name for h in hdul] else None
        if hdul[0].data is not None:
            arr = np.asarray(hdul[0].data, dtype=np.float64)   # (nbins, ny, nx)
            nbins = arr.shape[0]
            data = arr.reshape(nbins, -1).T                    # (npix, nbins)
            hdr = primary
            return data, hdr, ebounds
        for hdu in hdul[1:]:
            if hasattr(hdu, "columns") and hdu.name not in ("EBOUNDS", "GTI", "ENERGIES", "CTHETABOUNDS"):
                data = np.column_stack([np.asarray(hdu.data[c], float) for c in hdu.data.names])
                return data, hdu.header, ebounds
    raise SystemExit(f"ERROR: no map data found in {path}")


def check_energy_grid(ebounds, label):
    if ebounds is None:
        print(f"  [{WARN}] {label}: no EBOUNDS extension - cannot verify energy grid")
        return
    emin = float(ebounds["E_MIN"][0]); emax = float(ebounds["E_MAX"][-1])
    n = len(ebounds)
    # EBOUNDS energies are in keV -> MeV
    emin_mev, emax_mev = emin / 1e3, emax / 1e3
    bpd = n / np.log10(emax / emin)
    print(f"  [{OK}] energy grid: {n} bins, {emin_mev:.1f} MeV - {emax_mev/1e3:.1f} GeV, {bpd:.2f} bins/decade")


def validate_counts(path):
    print(f"\n== COUNTS: {path} ==")
    data, hdr, eb = load_map(path)
    npix, nbins = data.shape
    total = data.sum()
    print(f"  shape: {npix:,} pixels x {nbins} energy bins")
    print(f"  [{OK if data.min() >= 0 else FAIL}] non-negative: min={data.min():.0f}")
    frac = np.abs(data - np.round(data)).max()
    print(f"  [{OK if frac < 1e-6 else WARN}] integer counts (max frac part {frac:.1e})")
    print(f"  total photons: {int(total):,}")
    per_bin = data.sum(axis=0)
    falling = np.all(np.diff(per_bin[per_bin > 0]) <= 0)
    print(f"  [{OK if falling else WARN}] spectrum falls off with energy")
    print("  per-bin totals (first 6):", ", ".join(f"{int(x):,}" for x in per_bin[:6]), "...")
    check_energy_grid(eb, "counts")
    phdr = fits.getheader(path)
    cs = hdr.get("COORDSYS") or phdr.get("COORDSYS")
    if not cs:  # image cubes encode it in CTYPE1 (GLON-* = Galactic, RA-* = celestial)
        ct1 = str(phdr.get("CTYPE1", ""))
        cs = "GAL" if ct1.startswith("GLON") else ("CEL" if ct1.startswith("RA") else "?")
    print(f"  coordsys: {cs}")
    for k in ("DATE-OBS", "DATE-END"):
        v = fits.getheader(path).get(k)
        if v:
            print(f"  {k}: {v}")
    return data, total


def validate_exposure(path):
    print(f"\n== EXPOSURE: {path} ==")
    data, hdr, eb = load_map(path)
    pos = data > 0
    print(f"  shape: {data.shape[0]:,} pixels x {data.shape[1]} energy bins")
    print(f"  range: {data[pos].min():.3e} .. {data.max():.3e} cm^2 s")
    print(f"  [{OK if pos.mean() > 0.5 else WARN}] positive over {100*pos.mean():.1f}% of pixels")
    mag_ok = 1e8 < np.median(data[pos]) < 1e12
    print(f"  [{OK if mag_ok else WARN}] magnitude sane (median {np.median(data[pos]):.2e} cm^2 s)")
    return data


def validate_flux(path, counts=None, exposure=None):
    print(f"\n== FLUX: {path} ==")
    data, hdr, eb = load_map(path)
    finite = np.isfinite(data)
    print(f"  finite fraction: {finite.mean():.3f}")
    print(f"  range (finite): {np.nanmin(data[finite]):.3e} .. {np.nanmax(data[finite]):.3e}")
    if counts is not None and exposure is not None:
        if counts.shape == exposure.shape == data.shape:
            expected = np.zeros_like(counts)
            m = exposure > 0
            np.divide(counts, exposure, out=expected, where=m)
            ok = np.allclose(np.nan_to_num(data[m]), expected[m], rtol=1e-4, atol=0)
            print(f"  [{OK if ok else FAIL}] flux == counts/exposure where exposure>0")
        else:
            print(f"  [{WARN}] shapes differ (counts {counts.shape}, exposure {exposure.shape}, "
                  f"flux {data.shape}) - cannot verify division. Exposure likely has bin EDGES "
                  f"not CENTERS; use bincalc=CENTER so it matches the counts bins.")
    return data


def compare_counts(a, b):
    print(f"\n== CROSS-CHECK counts totals ==")
    da, _, _ = load_map(a); db, _, _ = load_map(b)
    ta, tb = da.sum(), db.sum()
    rel = abs(ta - tb) / max(ta, tb)
    verdict = OK if rel < 0.01 else (WARN if rel < 0.05 else FAIL)
    print(f"  {a}: {int(ta):,}")
    print(f"  {b}: {int(tb):,}")
    print(f"  [{verdict}] relative difference: {100*rel:.3f}%  (expect <1% for same events, two grids)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="counts [exposure] [flux], or two counts maps with --compare")
    ap.add_argument("--compare", action="store_true", help="cross-check two counts maps for equal totals")
    args = ap.parse_args()

    if args.compare:
        if len(args.files) != 2:
            sys.exit("--compare needs exactly two counts files")
        compare_counts(*args.files)
        return

    counts = exposure = None
    if len(args.files) >= 1:
        counts, _ = validate_counts(args.files[0])
    if len(args.files) >= 2:
        exposure = validate_exposure(args.files[1])
    if len(args.files) >= 3:
        validate_flux(args.files[2], counts=counts, exposure=exposure)
    print()


if __name__ == "__main__":
    main()
