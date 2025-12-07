#!/bin/bash
# Setup OSRM data for Korea/Daejeon region
# Idempotent: skips steps if files already exist

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OSRM_DATA_DIR="$PROJECT_DIR/osrm-data"
OSM_FILE="$OSRM_DATA_DIR/korea-latest.osm.pbf"
OSRM_FILE="$OSRM_DATA_DIR/korea-latest.osrm"

echo "=========================================="
echo "OSRM Setup for Korea/Daejeon"
echo "=========================================="
echo ""

# Create data directory
mkdir -p "$OSRM_DATA_DIR"

# Check if OSRM data already exists
if [ -f "$OSRM_FILE" ] && [ -s "$OSRM_FILE" ]; then
    echo "✓ OSRM data already exists at $OSRM_FILE"
    echo "  Skipping setup. To re-process, delete the file first."
    exit 0
fi

# Step 1: Download OSM data (if missing)
if [ -f "$OSM_FILE" ] && [ -s "$OSM_FILE" ]; then
    echo "✓ OSM file already exists: $(du -h "$OSM_FILE" | cut -f1)"
    echo "  Skipping download."
else
    echo "Step 1: Downloading South Korea OSM extract..."
    echo "  This may take 5-10 minutes..."
    curl -L --progress-bar -o "$OSM_FILE" \
        https://download.geofabrik.de/asia/south-korea-latest.osm.pbf
    
    if [ ! -f "$OSM_FILE" ] || [ ! -s "$OSM_FILE" ]; then
        echo "✗ Download failed"
        exit 1
    fi
    echo "✓ Download complete: $(du -h "$OSM_FILE" | cut -f1)"
fi
echo ""

# Step 2: Extract road network (if .osrm file missing OR datasource_names missing)
DATASOURCE_NAMES_FILE="$OSRM_DATA_DIR/korea-latest.osrm.datasource_names"
if [ -f "$OSRM_FILE" ] && [ -s "$OSRM_FILE" ] && [ -f "$DATASOURCE_NAMES_FILE" ] && [ -s "$DATASOURCE_NAMES_FILE" ]; then
    echo "✓ OSRM data already extracted (including datasource_names)"
    echo "  Skipping extraction."
else
    if [ -f "$OSRM_FILE" ] && [ ! -f "$DATASOURCE_NAMES_FILE" ]; then
        echo "⚠ OSRM file exists but datasource_names is missing"
        echo "  Re-extracting to regenerate required files..."
        # Remove existing .osrm files to force clean re-extraction
        rm -f "$OSRM_DATA_DIR"/korea-latest.osrm*
    fi
    
    echo "Step 2: Extracting road network (5-10 minutes)..."
    docker run --rm \
        -v "$OSRM_DATA_DIR:/data" \
        --platform linux/amd64 \
        osrm/osrm-backend:latest \
        osrm-extract -p /opt/car.lua /data/korea-latest.osm.pbf
    
    if [ ! -f "$OSRM_DATA_DIR/korea-latest.osrm" ]; then
        echo "✗ Extraction failed"
        exit 1
    fi
    
    if [ ! -f "$DATASOURCE_NAMES_FILE" ]; then
        echo "✗ Extraction incomplete: datasource_names file missing"
        exit 1
    fi
    
    echo "✓ Extraction complete"
fi
echo ""

# Step 3: Contract graph (if already done, skip)
if [ -f "$OSRM_FILE" ] && [ -s "$OSRM_FILE" ]; then
    # Check if all required .osrm files exist
    required_files=(
        "$OSRM_DATA_DIR/korea-latest.osrm"
        "$OSRM_DATA_DIR/korea-latest.osrm.ebg"
        "$OSRM_DATA_DIR/korea-latest.osrm.edges"
    )
    all_exist=true
    for f in "${required_files[@]}"; do
        if [ ! -f "$f" ] || [ ! -s "$f" ]; then
            all_exist=false
            break
        fi
    done
    
    if [ "$all_exist" = true ]; then
        echo "✓ OSRM data already contracted"
        echo "  Skipping contraction."
    else
        echo "Step 3: Contracting graph (2-5 minutes)..."
        docker run --rm \
            -v "$OSRM_DATA_DIR:/data" \
            --platform linux/amd64 \
            osrm/osrm-backend:latest \
            osrm-contract /data/korea-latest.osrm
        echo "✓ Contraction complete"
    fi
else
    echo "Step 3: Contracting graph (2-5 minutes)..."
    docker run --rm \
        -v "$OSRM_DATA_DIR:/data" \
        --platform linux/amd64 \
        osrm/osrm-backend:latest \
        osrm-contract /data/korea-latest.osrm
    echo "✓ Contraction complete"
fi
echo ""

# Verify final result
if [ -f "$OSRM_FILE" ] && [ -s "$OSRM_FILE" ]; then
    echo "=========================================="
    echo "✓ OSRM setup complete!"
    echo "=========================================="
    echo ""
    echo "OSRM data ready at: $OSRM_FILE"
    echo ""
    echo "Next: docker compose up -d"
else
    echo "✗ Setup failed: OSRM file not found"
    exit 1
fi
