#!/usr/bin/env python3

import numpy as np
import healpy as hp
import cosmoglobe
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.units import Unit

# -----------------------------
# USER PARAMETERS
# -----------------------------
COUNTS_FITS   = "data/lat_source_zmax90_gt1gev_ccube_merged.fits"
EXPOSURE_FITS = "data/lat_source_zmax90_gt1gev_expcube1_merged.fits"

OUT_RESIDUAL  = "residual_sync_subtracted_intensity.fits"
OUT_MODEL     = "bestfit_sync_model_intensity.fits"
OUT_INTENSITY = "intensity.fits"

# Synchrotron frequency (morphology only matters)
SYNC_FREQ_GHZ = 30.0

FWHM_DEG = 1.                   # Smoothing scale for synchrotron template (degrees)
ABS_B_MIN_DEG = 10.0            # Mask out Galactic plane (|b| < ABS_B_MIN_DEG)
MIN_COUNTS_FOR_WEIGHT = 1.0     # Minimum counts to avoid infinite weights in Poisson fitting

# HEALPix nside for generating synchrotron template
NSIDE = 512                     # High enough to capture small-scale structure, but not too high to be slow. Adjust as needed.

# -----------------------------
# Helper function to convert HEALPix to CAR projection
# -----------------------------
def healpix_to_car(healpix_map, nlat, nlon):
    """
    Convert a HEALPix map to CAR (Cartesian) projection.

    Parameters:
    -----------
    healpix_map : ndarray
        HEALPix format map
    nlat : int
        Number of latitude pixels
    nlon : int
        Number of longitude pixels

    Returns:
    --------
    car_map : ndarray with shape (nlat, nlon)
        Data in CAR projection
    """
    # Create coordinate grids for CAR projection
    # Latitude: -90 to +90, Longitude: -180 to +180
    lat_centers = np.linspace(-90, 90, nlat)
    lon_centers = np.linspace(-180, 180, nlon)

    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

    # Convert to theta, phi for HEALPix
    theta = np.deg2rad(90 - lat_grid)  # Colatitude
    phi = np.deg2rad(lon_grid)
    phi[phi < 0] += 2 * np.pi  # Shift to [0, 2π]

    # Get HEALPix nside from map length
    nside = hp.npix2nside(len(healpix_map))

    # Sample HEALPix map at CAR grid positions
    car_map = hp.get_interp_val(healpix_map, theta.flatten(), phi.flatten())
    car_map = car_map.reshape(nlat, nlon)

    return car_map

# -----------------------------
# Load Fermi maps (CAR projection)
# -----------------------------
print("Loading Fermi data...")
with fits.open(COUNTS_FITS) as hdul:
    counts = hdul[0].data[0, :, :]  # First energy bin
    counts_header = hdul[0].header

with fits.open(EXPOSURE_FITS) as hdul:
    exposure = hdul[0].data[0, :, :]  # First energy bin

nlat, nlon = counts.shape
print(f"Data shape: {nlat} x {nlon} (lat x lon)")

# Compute intensity map
print("Computing intensity map...")
mask_valid = exposure > 0
intensity = np.zeros_like(counts, dtype=np.float64)
intensity[mask_valid] = counts[mask_valid] / exposure[mask_valid]

# -----------------------------
# Generate Cosmoglobe synchrotron template
# -----------------------------
print(f"Generating Cosmoglobe synchrotron template at {SYNC_FREQ_GHZ} GHz...")
model = cosmoglobe.sky_model(nside=NSIDE)

# Evaluate synchrotron at chosen frequency (returns HEALPix)
uK_RJ = Unit('uK_RJ')
sync_stokes = model.components["synch"].get_delta_emission(
    SYNC_FREQ_GHZ * u.GHz,
    output_unit=uK_RJ
)

# Extract intensity (Stokes I) from HEALPix
if hasattr(sync_stokes, "value"):
    sync_hp = sync_stokes.value[0, :]
else:
    sync_hp = sync_stokes[0, :]

# Smooth in HEALPix space
print(f"Smoothing template to {FWHM_DEG} deg FWHM...")
sync_hp = hp.smoothing(sync_hp, fwhm=np.deg2rad(FWHM_DEG))

# Convert synchrotron template to CAR projection
print("Converting synchrotron template to CAR projection...")
sync = healpix_to_car(sync_hp, nlat, nlon)

# -----------------------------
# Create coordinate grids and masks
# -----------------------------
print(f"Creating Galactic plane mask (|b| >= {ABS_B_MIN_DEG} deg)...")
lat_centers = np.linspace(-90, 90, nlat)
lon_centers = np.linspace(-180, 180, nlon)
lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)

# Galactic latitude mask (assuming data is already in Galactic coordinates)
mask_plane = np.abs(lat_grid) >= ABS_B_MIN_DEG
mask = mask_valid & mask_plane

npix_total = nlat * nlon
npix_fit = np.sum(mask)
print(f"Number of pixels in fit: {npix_fit} / {npix_total}")

# Flatten arrays for fitting
d = intensity[mask]
T0 = np.ones_like(d)
T1 = sync[mask]
# Design matrix
T0 = np.ones_like(d)
T1 = sync[mask]
T = np.vstack([T0, T1]).T

# -----------------------------
# Poisson weights
# Var(I) ≈ C / E^2
# -----------------------------
print("Fitting synchrotron template...")
C = counts[mask]
E = exposure[mask]

C_safe = np.maximum(C, MIN_COUNTS_FOR_WEIGHT)
w = (E * E) / C_safe

# Weighted GLS
Tw = T * w[:, None]
A = Tw.T @ T
bvec = Tw.T @ d

a_hat = np.linalg.solve(A, bvec)
a0, a_sync = a_hat

print("\nBest-fit coefficients:")
print(f"  Offset (a0):        {a0:.6e}")
print(f"  Synchrotron (a_sync): {a_sync:.6e}")

# -----------------------------
# Build residual and model maps
# -----------------------------
print("\nBuilding model and residual maps...")
model_sync = a_sync * sync
residual = intensity - model_sync

# Set masked regions to NaN for cleaner visualization
residual_clean = residual.copy()
model_clean = model_sync.copy()
intensity_clean = intensity.copy()
residual_clean[~mask_valid] = np.nan
model_clean[~mask_valid] = np.nan
intensity_clean[~mask_valid] = np.nan

# -----------------------------
# Evaluate foreground removal effectiveness
# -----------------------------
print("\n=== Foreground Removal Assessment ===")
print("Comparing intensity before and after synchrotron subtraction:")
print("(Lower values = closer to zero = better foreground removal)\n")

# Use the fitted region for comparison
fit_mask = mask_valid & mask_plane

intensity_fit = intensity[fit_mask]
residual_fit = residual[fit_mask]

# Compute metrics
rms_before = np.sqrt(np.mean(intensity_fit**2))
rms_after = np.sqrt(np.mean(residual_fit**2))

mean_abs_before = np.mean(np.abs(intensity_fit))
mean_abs_after = np.mean(np.abs(residual_fit))

median_abs_before = np.median(np.abs(intensity_fit))
median_abs_after = np.median(np.abs(residual_fit))

# Count pixels improved (closer to zero)
improved = np.abs(residual_fit) < np.abs(intensity_fit)
n_improved = np.sum(improved)
n_total = len(improved)

print(f"RMS (Root Mean Square):")
print(f"  Before: {rms_before:.6e}")
print(f"  After:  {rms_after:.6e}")
print(f"  Change: {100*(rms_after - rms_before)/rms_before:+.1f}%")

print(f"\nMean Absolute Value:")
print(f"  Before: {mean_abs_before:.6e}")
print(f"  After:  {mean_abs_after:.6e}")
print(f"  Change: {100*(mean_abs_after - mean_abs_before)/mean_abs_before:+.1f}%")

print(f"\nMedian Absolute Value:")
print(f"  Before: {median_abs_before:.6e}")
print(f"  After:  {median_abs_after:.6e}")
print(f"  Change: {100*(median_abs_after - median_abs_before)/median_abs_before:+.1f}%")

print(f"\nPixels Improved (closer to zero):")
print(f"  {n_improved} / {n_total} ({100*n_improved/n_total:.1f}%)")

if rms_after < rms_before:
    print(f"\n✓ Synchrotron removal IMPROVES the map (reduces RMS by {100*(1-rms_after/rms_before):.1f}%)")
else:
    print(f"\n✗ Synchrotron removal WORSENS the map (increases RMS by {100*(rms_after/rms_before-1):.1f}%)")


# -----------------------------
# Save output FITS files in CAR projection
# -----------------------------
print("Writing output maps...")

# Create HDU with proper header
hdu_residual = fits.PrimaryHDU(data=residual_clean[np.newaxis, :, :], header=counts_header)
hdu_model = fits.PrimaryHDU(data=model_clean[np.newaxis, :, :], header=counts_header)
hdu_intensity = fits.PrimaryHDU(data=intensity_clean[np.newaxis, :, :], header=counts_header)

# Update headers
hdu_residual.header['COMMENT'] = 'Intensity with synchrotron subtracted'
hdu_model.header['COMMENT'] = f'Best-fit synchrotron model at {SYNC_FREQ_GHZ} GHz'
hdu_intensity.header['COMMENT'] = 'Gamma-ray intensity (counts/exposure)'

# Write files
hdu_residual.writeto(OUT_RESIDUAL, overwrite=True)
hdu_model.writeto(OUT_MODEL, overwrite=True)
hdu_intensity.writeto(OUT_INTENSITY, overwrite=True)

print(f"Saved: {OUT_RESIDUAL}")
print(f"Saved: {OUT_MODEL}")
print(f"Saved: {OUT_INTENSITY}")
print("Done.")