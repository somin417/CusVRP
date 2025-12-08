#!/bin/bash
# Complete test script: OSRM → VROOM → Run 20-node experiment
# Usage: ./run_test_20nodes.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "VRP Test Script - 20 Nodes"
echo "=========================================="
echo ""

# Step 1: Activate virtual environment
echo "Step 1: Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "✗ Virtual environment not found at ./venv"
    echo "  Create it with: python3 -m venv venv"
    exit 1
fi
echo ""

# Step 2: Check OSRM data
echo "Step 2: Checking OSRM data..."
OSRM_FILEINDEX="./osrm-data/korea-latest.osrm.fileIndex"
if [ -f "$OSRM_FILEINDEX" ] && [ -s "$OSRM_FILEINDEX" ]; then
    echo "✓ OSRM data exists: $(du -h "$OSRM_FILEINDEX" | cut -f1)"
else
    echo "✗ OSRM data not found at $OSRM_FILEINDEX"
    echo "  Run: ./install_native_services.sh"
    exit 1
fi
echo ""

# Step 3: Check services
echo "Step 3: Checking services..."
OSRM_RUNNING=$(pgrep -f "osrm-routed" > /dev/null && echo "yes" || echo "no")
VROOM_RUNNING=$(pgrep -f "vroom_server.py" > /dev/null && echo "yes" || echo "no")

if [ "$OSRM_RUNNING" != "yes" ] || [ "$VROOM_RUNNING" != "yes" ]; then
    echo "⚠ Services not running. Starting services..."
    ./start_services.sh
    sleep 5
fi
echo "✓ Services are running"
echo ""

# Step 4: Test OSRM endpoint
echo "Step 4: Testing OSRM endpoint..."
OSRM_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://localhost:5001/route/v1/driving/127.385,36.35;127.386,36.36" || echo "000")

if [ "$OSRM_RESPONSE" = "200" ]; then
    echo "✓ OSRM is responding (HTTP $OSRM_RESPONSE)"
else
    echo "✗ OSRM not responding (HTTP $OSRM_RESPONSE)"
    echo "  Check: cat osrm.log"
    exit 1
fi
echo ""

# Step 5: Test VROOM endpoint
echo "Step 5: Testing VROOM endpoint..."
VROOM_TEST='{"vehicles":[{"id":1,"start":[127.385,36.35],"end":[127.385,36.35],"capacity":[100]}],"jobs":[{"id":1,"location":[127.386,36.36],"amount":[10]}]}'

VROOM_RESPONSE=$(curl -s -X POST http://localhost:3000/ \
    -H "Content-Type: application/json" \
    -d "$VROOM_TEST" \
    -o /dev/null -w "%{http_code}" || echo "000")

if [ "$VROOM_RESPONSE" = "200" ]; then
    echo "✓ VROOM is responding (HTTP $VROOM_RESPONSE)"
else
    echo "✗ VROOM not responding (HTTP $VROOM_RESPONSE)"
    echo "  Check: cat vroom.log"
    exit 1
fi
echo ""

# Step 6: Run experiment with 20 nodes
echo "=========================================="
echo "Step 6: Running 20-node experiment"
echo "=========================================="
echo ""

python -m src.vrp_fairness.run_experiment \
    --seed 0 \
    --n 20 \
    --city daejeon \
    --dcs "36.35,127.385" \
    --eps 0.10 \
    --iters 300

EXPERIMENT_EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
    echo "✓ Experiment completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved in: outputs/"
    echo "  - baseline.json"
    echo "  - improved.json"
    echo "  - comparison.csv"
    echo "  - plots/"
else
    echo "✗ Experiment failed (exit code: $EXPERIMENT_EXIT_CODE)"
    echo "=========================================="
    exit $EXPERIMENT_EXIT_CODE
fi

