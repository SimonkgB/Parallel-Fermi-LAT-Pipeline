#!/usr/bin/env python3
"""Divide two FITS maps: flux = counts / exposure.

Replacement for `farith DIV`, which only operates on image HDUs and cannot
handle HEALPix maps (stored as binary tables, e.g. gtbin HEALPIX output).
Works for both image maps and HEALPix maps.

Pixels with zero exposure are set to 0 in the output.

Usage:
    python3 scripts/hpx_divide.py counts.fits exposure.fits flux.fits
"""
import sys
import numpy as np
from astropy.io import fits


def find_map_data(hdul):
    """Return (ext_index, 2D/3D float array) of the map in a FITS file.

    Image maps: primary HDU. HEALPix maps: first binary-table extension
    that isn't metadata (EBOUNDS/GTI/ENERGIES), one column per energy bin.
    """
    if hdul[0].data is not None:
        return 0, np.asarray(hdul[0].data, dtype=np.float64)
    for i, hdu in enumerate(hdul):
        if i == 0 or hdu.data is None or not hasattr(hdu, 'columns'):
            continue
        if hdu.name in ('EBOUNDS', 'GTI', 'ENERGIES'):
            continue
        arr = np.column_stack([hdu.data[name] for name in hdu.data.names]).astype(np.float64)
        return i, arr
    sys.exit(f"ERROR: no image or table map data found")


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    counts_path, exposure_path, out_path = sys.argv[1:4]

    counts_hdul = fits.open(counts_path)
    c_ext, counts = find_map_data(counts_hdul)
    with fits.open(exposure_path) as hdul:
        _, exposure = find_map_data(hdul)

    if counts.shape != exposure.shape:
        # gtexpcube2 with bincalc=EDGE emits N+1 energy planes (bin edges) for
        # N counts bins; convert edges -> bin centers via geometric mean.
        # Energy is the last axis for HEALPix tables, the first for image cubes.
        if counts.ndim == 2 and exposure.shape == (counts.shape[0], counts.shape[1] + 1):
            print(f"Exposure has {exposure.shape[1]} energy planes vs {counts.shape[1]} counts bins: "
                  "converting edges -> centers (geometric mean)")
            exposure = np.sqrt(exposure[:, :-1] * exposure[:, 1:])
        elif counts.ndim >= 2 and exposure.shape[0] == counts.shape[0] + 1 and exposure.shape[1:] == counts.shape[1:]:
            print(f"Exposure has {exposure.shape[0]} energy planes vs {counts.shape[0]} counts bins: "
                  "converting edges -> centers (geometric mean)")
            exposure = np.sqrt(exposure[:-1] * exposure[1:])
        else:
            sys.exit(f"ERROR: shape mismatch: counts {counts.shape} vs exposure {exposure.shape}")

    flux = np.zeros_like(counts)
    np.divide(counts, exposure, out=flux, where=exposure > 0)

    # Write flux into a copy of the counts file (same structure/metadata)
    if c_ext == 0:
        counts_hdul[0].data = flux
    else:
        table = counts_hdul[c_ext].data
        # Column dtypes are often int for counts; rebuild columns as float
        cols = [fits.Column(name=name, format='D', array=flux[:, j])
                for j, name in enumerate(table.names)]
        new_hdu = fits.BinTableHDU.from_columns(cols, header=counts_hdul[c_ext].header, name=counts_hdul[c_ext].name)
        counts_hdul[c_ext] = new_hdu
    counts_hdul[0].header['HISTORY'] = f"flux = {counts_path} / {exposure_path} (hpx_divide.py)"
    counts_hdul.writeto(out_path, overwrite=True)
    print(f"Created flux map: {out_path}")


if __name__ == "__main__":
    main()
