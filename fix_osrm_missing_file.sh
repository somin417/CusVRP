#!/bin/bash
# Quick fix: Re-extract OSRM data to generate missing datasource_names file

set -e

cd "$(dirname "$0")"
OSRM_DATA_DIR="./osrm-data"
OSM_FILE="$OSRM_DATA_DIR/korea-latest.osm.pbf"
OSRM_FILE="$OSRM_DATA_DIR/korea-latest.osrm"
DATASOURCE_NAMES_FILE="$OSRM_DATA_DIR/korea-latest.osrm.datasource_names"

echo "=========================================="
echo "Fix: Regenerating OSRM datasource_names"
echo "=========================================="
echo ""

# Check if OSM file exists
if [ ! -f "$OSM_FILE" ] || [ ! -s "$OSM_FILE" ]; then
    echo "✗ OSM file not found: $OSM_FILE"
    echo "  Run: ./scripts/setup_osrm_korea.sh"
    exit 1
fi

# Remove existing .osrm files (but keep .osm.pbf)
echo "Removing existing .osrm files to force re-extraction..."
rm -f "$OSRM_DATA_DIR"/korea-latest.osrm*

echo "Re-extracting road network (5-10 minutes)..."
docker run --rm \
    -v "$OSRM_DATA_DIR:/data" \
    --platform linux/amd64 \
    osrm/osrm-backend:latest \
    osrm-extract -p /opt/car.lua /data/korea-latest.osm.pbf

if [ ! -f "$OSRM_FILE" ]; then
    echo "✗ Extraction failed"
    exit 1
fi

if [ ! -f "$DATASOURCE_NAMES_FILE" ]; then
    echo "✗ Extraction incomplete: datasource_names still missing"
    exit 1
fi

echo "✓ Extraction complete - datasource_names file created"
echo ""

# Re-contract
echo "Re-contracting graph (2-5 minutes)..."
docker run --rm \
    -v "$OSRM_DATA_DIR:/data" \
    --platform linux/amd64 \
    osrm/osrm-backend:latest \
    osrm-contract /data/korea-latest.osrm

echo ""
echo "=========================================="
echo "✓ Fix complete! OSRM data regenerated"
echo "=========================================="
echo ""
echo "Now run: docker compose up -d"

