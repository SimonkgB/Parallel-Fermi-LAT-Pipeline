#!/bin/bash

# Quick diagnostic to find where data is being lost
eval "$(micromamba shell hook --shell bash)"
micromamba activate fermi

echo "Running diagnostic on first 2 weekly files..."

PHOTON_DIR="../weekly/photon/"
OUT_DIR="./diagnostic_test"
mkdir -p "$OUT_DIR"

# Get first 2 weekly files
files=($(ls $PHOTON_DIR/lat_photon_weekly_w*.fits | head -2))

echo "Processing ${#files[@]} files:"
for f in "${files[@]}"; do
    echo "  $f"
done
echo ""

total_events=0

for i in "${!files[@]}"; do
    input_file="${files[$i]}"

    echo "================================================"
    echo "File $((i+1)): $(basename $input_file)"
    echo "================================================"

    # Count events in raw file
    python3 << EOF
from astropy.io import fits
with fits.open("$input_file") as h:
    n = len(h['EVENTS'].data)
    print(f"Raw file: {n:,} events")
EOF

    # First gtselect
    temp1="$OUT_DIR/temp1_$i.fits"
    gtselect evclass=INDEF evtype=INDEF \
        infile="$input_file" \
        outfile="$temp1" \
        ra=0 dec=0 rad=180 \
        tmin=INDEF tmax=INDEF \
        emin=30 emax=1000000 \
        zmax=180 > /dev/null 2>&1

    python3 << EOF
from astropy.io import fits
with fits.open("$temp1") as h:
    n = len(h['EVENTS'].data)
    print(f"After gtselect #1 (E>30MeV): {n:,} events")
EOF

    # Second gtselect (energy filter)
    temp2="$OUT_DIR/temp2_$i.fits"
    gtselect \
        infile="$temp1" \
        outfile="$temp2" \
        ra=0 dec=0 rad=180 \
        tmin=INDEF tmax=INDEF \
        emin=1000 emax=500000 \
        zmax=90 > /dev/null 2>&1

    python3 << EOF
from astropy.io import fits
with fits.open("$temp2") as h:
    n = len(h['EVENTS'].data)
    print(f"After gtselect #2 (E>1GeV, zmax<90): {n:,} events")
EOF

    # GTI filter
    temp3="$OUT_DIR/temp3_$i.fits"
    gtmktime \
        scfile=../mission/spacecraft/lat_spacecraft_merged.fits \
        filter="(DATA_QUAL>0)&&(LAT_CONFIG==1)" \
        roicut=no \
        evfile="$temp2" \
        outfile="$temp3" > /dev/null 2>&1

    events_final=$(python3 << EOF
from astropy.io import fits
with fits.open("$temp3") as h:
    n = len(h['EVENTS'].data)
    print(f"After gtmktime (GTI): {n:,} events")
    print(n)
EOF
)

    events_final=$(echo "$events_final" | tail -1)
    total_events=$((total_events + events_final))

    echo ""
done

echo "================================================"
echo "TOTAL EVENTS FROM 2 FILES: $total_events"
echo "================================================"
echo ""
echo "With 913 weekly files, expected total would be approximately:"
echo "  $((total_events * 913 / 2)) events"
echo ""
echo "Compare this to your single-processor result of 42,683,454 events"

# Clean up
rm -rf "$OUT_DIR"
