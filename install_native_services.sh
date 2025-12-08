#!/bin/bash
# Install VROOM and OSRM natively (without Docker)
# Run this script to install both services

set -e

echo "=========================================="
echo "Installing VROOM and OSRM (Native)"
echo "=========================================="
echo ""

# Check if running as root for install steps
if [ "$EUID" -eq 0 ]; then 
    SUDO=""
else
    SUDO="sudo"
fi

# Step 1: Install dependencies
echo "Step 1: Installing build dependencies..."
$SUDO apt-get update
$SUDO apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    libbz2-dev \
    libstxxl-dev \
    libstxxl1v5 \
    libxml2-dev \
    libzip-dev \
    liblua5.2-dev \
    libtbb-dev \
    pkg-config \
    git \
    curl \
    wget \
    screen

echo "✓ Dependencies installed"
echo ""

# Step 2: Install OSRM
echo "Step 2: Installing OSRM..."
if command -v osrm-routed &> /dev/null; then
    echo "✓ OSRM already installed"
else
    BUILD_DIR="$HOME/osrm-build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    if [ ! -d "osrm-backend" ]; then
        echo "  Cloning OSRM repository..."
        git clone https://github.com/Project-OSRM/osrm-backend.git
    fi
    
    cd osrm-backend
    echo "  Building OSRM (this may take 10-20 minutes)..."
    mkdir -p build
    cd build
    # Fix for GCC 13 strict array bounds checking
    cmake .. -DCMAKE_CXX_FLAGS="-Wno-error=array-bounds"
    cmake --build .
    $SUDO cmake --build . --target install
    
    echo "✓ OSRM installed"
fi
echo ""

# Step 3: Install VROOM
echo "Step 3: Installing VROOM..."
if command -v vroom &> /dev/null; then
    echo "✓ VROOM already installed"
else
    BUILD_DIR="$HOME/vroom-build"
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    
    if [ ! -d "vroom" ]; then
        echo "  Cloning VROOM repository..."
        git clone https://github.com/VROOM-Project/vroom.git
        cd vroom
        echo "  Initializing git submodules..."
        git submodule update --init --recursive
        cd ..
    fi
    
    cd vroom
    echo "  Building VROOM (this may take 5-10 minutes)..."
    
    # Install VROOM dependencies
    $SUDO apt-get install -y libasio-dev rapidjson-dev
    
    # Build from src directory
    cd src
    make
    cd ..
    
    # Install (copy binaries)
    $SUDO mkdir -p /usr/local/bin
    $SUDO cp bin/vroom /usr/local/bin/
    
    echo "✓ VROOM installed"
fi
echo ""

# Step 4: Process OSRM data (if needed)
echo "Step 4: Processing OSRM data..."
cd /home/comet/CusVRP

OSRM_DATA_DIR="./osrm-data"
OSM_FILE="$OSRM_DATA_DIR/korea-latest.osm.pbf"
OSRM_FILE="$OSRM_DATA_DIR/korea-latest.osrm"
OSRM_FILEINDEX="$OSRM_DATA_DIR/korea-latest.osrm.fileIndex"

if [ ! -f "$OSRM_FILEINDEX" ] || [ ! -f "$OSRM_DATA_DIR/korea-latest.osrm.datasource_names" ] || [ ! -f "$OSRM_DATA_DIR/korea-latest.osrm.partition" ]; then
    if [ ! -f "$OSM_FILE" ]; then
        echo "✗ OSM file not found: $OSM_FILE"
        echo "  Run: ./scripts/setup_osrm_korea.sh (download step only)"
        exit 1
    fi
    
    echo "  Extracting OSRM data (5-10 minutes)..."
    cd "$OSRM_DATA_DIR"
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH osrm-extract -p /usr/local/share/osrm/profiles/car.lua "$(basename "$OSM_FILE")"
    
    echo "  Partitioning OSRM data (2-3 minutes)..."
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH osrm-partition "$(basename "$OSRM_FILE")"
    
    echo "  Customizing OSRM data (2-3 minutes)..."
    LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH osrm-customize "$(basename "$OSRM_FILE")"
    cd "$PROJECT_DIR"
    
    echo "✓ OSRM data processed"
else
    echo "✓ OSRM data already processed"
fi
echo ""

# Step 5: Create startup script
echo "Step 5: Creating startup script..."
cat > start_services.sh << 'EOF'
#!/bin/bash
# Start OSRM and VROOM services

cd /home/comet/CusVRP

# Kill existing processes
pkill -f "osrm-routed" || true
pkill -f "vroom" || true
sleep 2

# Start OSRM
echo "Starting OSRM on port 5001..."
osrm-routed --algorithm mld -p 5001 osrm-data/korea-latest.osrm > osrm.log 2>&1 &
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

# Start VROOM
echo "Starting VROOM on port 3000..."
export OSRM_ADDRESS=http://localhost:5001
vroom -a $OSRM_ADDRESS -p 3000 > vroom.log 2>&1 &
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
EOF

chmod +x start_services.sh
echo "✓ Startup script created: ./start_services.sh"
echo ""

echo "=========================================="
echo "✓ Installation complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: ./start_services.sh"
echo "2. Test: curl http://localhost:3000/"
echo "3. Run your Python code (it will use localhost:3000)"
echo ""

