#!/bin/bash
# Always run from the repository root, no matter where the script is called from
cd "$(dirname "$0")/.." || exit 1

eval "$(micromamba shell hook --shell bash)"
micromamba activate fermi

echo "========================================"
echo "Event counts in data/ files"
echo "========================================"

python3 <<'EOF'
from astropy.io import fits
import numpy as np
import glob
import os

def total_counts(hdul):
    """Sum all photons, whether stored as an image cube (CCUBE, primary HDU)
    or a HEALPix map (SKYMAP binary table, one column per energy bin)."""
    if hdul[0].data is not None:
        return int(np.sum(hdul[0].data))
    for hdu in hdul[1:]:
        if hasattr(hdu, "columns") and hdu.name not in ("EBOUNDS", "GTI", "ENERGIES", "CTHETABOUNDS"):
            return int(sum(np.sum(hdu.data[c]) for c in hdu.data.names))
    return 0

files = sorted(glob.glob("./data/*.fits"))

for filepath in files:
    filename = os.path.basename(filepath)
    try:
        with fits.open(filepath) as hdul:
            if "EVENTS" in [hdu.name for hdu in hdul]:
                # Event list file (GTI)
                count = len(hdul["EVENTS"].data)
                print(f"{filename:55s} {count:>12,} events")
            elif filename.startswith("counts"):
                # Counts cube - sum of all photons (image or HEALPix)
                count = total_counts(hdul)
                print(f"{filename:55s} {count:>12,} photons")
            elif filename.startswith("livetime") or filename.startswith("exposure"):
                # Livetime/exposure cubes - show file type only
                print(f"{filename:55s}      (exposure map)")
            elif filename.startswith("flux"):
                # Flux map (counts/exposure) - flux values, not counts
                print(f"{filename:55s}      (flux map)")
            else:
                print(f"{filename:55s}      (valid FITS)")
    except Exception as e:
        print(f"{filename:55s} ERROR: {e}")

print("========================================")
EOF
