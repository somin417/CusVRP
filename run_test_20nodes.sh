#!/bin/bash
# Complete test script: Docker → OSRM → VROOM → Run 20-node experiment
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

# Step 2: Check Docker
echo "Step 2: Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "✗ Docker is not running"
    echo "  Please start Docker Desktop and try again"
    exit 1
fi
echo "✓ Docker is running"
echo ""

# Step 3: Check OSRM data
echo "Step 3: Checking OSRM data..."
OSRM_FILE="./osrm-data/korea-latest.osrm"
if [ -f "$OSRM_FILE" ] && [ -s "$OSRM_FILE" ]; then
    echo "✓ OSRM data exists: $(du -h "$OSRM_FILE" | cut -f1)"
else
    echo "✗ OSRM data not found at $OSRM_FILE"
    echo "  Run: ./scripts/setup_osrm_korea.sh"
    exit 1
fi
echo ""

# Step 4: Check Docker containers
echo "Step 4: Checking Docker containers..."
docker compose ps

OSRM_STATUS=$(docker compose ps osrm 2>/dev/null | grep -q "Up" && echo "up" || echo "down")
VROOM_STATUS=$(docker compose ps vroom 2>/dev/null | grep -q "Up" && echo "up" || echo "down")

if [ "$OSRM_STATUS" != "up" ] || [ "$VROOM_STATUS" != "up" ]; then
    echo ""
    echo "⚠ Containers not running. Starting services..."
    docker compose up -d
    
    echo "Waiting for services to be healthy (30 seconds)..."
    sleep 30
    
    # Check again
    OSRM_STATUS=$(docker compose ps osrm 2>/dev/null | grep -q "Up" && echo "up" || echo "down")
    VROOM_STATUS=$(docker compose ps vroom 2>/dev/null | grep -q "Up" && echo "up" || echo "down")
    
    if [ "$OSRM_STATUS" != "up" ] || [ "$VROOM_STATUS" != "up" ]; then
        echo "✗ Services failed to start"
        echo "  Check logs: docker compose logs"
        exit 1
    fi
fi
echo "✓ Docker containers are running"
echo ""

# Step 5: Test OSRM endpoint
echo "Step 5: Testing OSRM endpoint..."
OSRM_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    "http://localhost:5001/route/v1/driving/127.385,36.35;127.386,36.36" || echo "000")

if [ "$OSRM_RESPONSE" = "200" ]; then
    echo "✓ OSRM is responding (HTTP $OSRM_RESPONSE)"
else
    echo "✗ OSRM not responding (HTTP $OSRM_RESPONSE)"
    echo "  Check: docker compose logs osrm"
    exit 1
fi
echo ""

# Step 6: Test VROOM endpoint
echo "Step 6: Testing VROOM endpoint..."
VROOM_TEST='{"vehicles":[{"id":1,"start":[127.385,36.35],"end":[127.385,36.35],"capacity":[100]}],"jobs":[{"id":1,"location":[127.386,36.36],"amount":[10]}]}'

VROOM_RESPONSE=$(curl -s -X POST http://localhost:3000/ \
    -H "Content-Type: application/json" \
    -d "$VROOM_TEST" \
    -o /dev/null -w "%{http_code}" || echo "000")

if [ "$VROOM_RESPONSE" = "200" ]; then
    echo "✓ VROOM is responding (HTTP $VROOM_RESPONSE)"
else
    echo "✗ VROOM not responding (HTTP $VROOM_RESPONSE)"
    echo "  Check: docker compose logs vroom"
    exit 1
fi
echo ""

# Step 7: Run experiment with 20 nodes
echo "=========================================="
echo "Step 7: Running 20-node experiment"
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

