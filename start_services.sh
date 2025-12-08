#!/bin/bash
# Start OSRM and VROOM services

cd /home/comet/CusVRP

# Kill existing processes
pkill -f "osrm-routed" || true
pkill -f "vroom" || true
sleep 2

# Start OSRM
echo "Starting OSRM on port 5001..."
# Use system libraries instead of anaconda's and fix Boost version
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH osrm-routed --algorithm mld -p 5001 osrm-data/korea-latest.osrm > osrm.log 2>&1 &
OSRM_PID=$!
echo "OSRM started with PID: $OSRM_PID"

# Wait for OSRM to be ready
echo "Waiting for OSRM to start..."
sleep 5

# Test OSRM
if curl -s "http://localhost:5001/route/v1/driving/127.0,36.0;127.1,36.1" > /dev/null; then
    echo "✓ OSRM is ready"
else
    echo "✗ OSRM failed to start. Check osrm.log"
    exit 1
fi

# Start VROOM HTTP server (Python wrapper)
echo "Starting VROOM HTTP server on port 3000..."
export OSRM_ADDRESS=http://localhost:5001
python3 /home/comet/CusVRP/vroom_server.py > vroom.log 2>&1 &
VROOM_PID=$!
echo "VROOM started with PID: $VROOM_PID"

# Wait for VROOM to be ready
echo "Waiting for VROOM to start..."
sleep 3

# Test VROOM
if curl -s http://localhost:3000/ > /dev/null; then
    echo "✓ VROOM is ready"
else
    echo "✗ VROOM failed to start. Check vroom.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ Services started successfully!"
echo "=========================================="
echo "OSRM: http://localhost:5001"
echo "VROOM: http://localhost:3000"
echo ""
echo "PIDs: OSRM=$OSRM_PID, VROOM=$VROOM_PID"
echo "Logs: osrm.log, vroom.log"
echo ""
echo "To stop: pkill -f 'osrm-routed|vroom'"
