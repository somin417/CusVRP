# VRP Fairness Optimization System

Python system for solving Vehicle Routing Problems (VRP) with fairness optimization. Uses VROOM for baseline routing and ALNS for improving fairness (waiting time distribution) while respecting cost budgets.

## Features

- **Multi-Depot VRP**: Assigns stops to nearest depot using OSRM travel time, then solves per-depot VRP
- **Fairness Optimization**: ALNS algorithm optimizes combined objective (Z = α·Z1/Z1* + β·Z2/Z2* + γ·Z3/Z3*)
  - Z1: Min-Max Weighted Waiting Time
  - Z2: Total Routing Cost (distance or time)
  - Z3: Weighted Variance of Waiting Time
- **Road Network Routing**: OSRM for accurate travel times, distances, and route geometry
- **Real Data Support**: Load stops from GeoPackage (GPKG) files
- **Interactive Maps**: Folium-based HTML maps with baseline vs improved route comparison
- **Reproducible**: Deterministic seeds and experiment logging

---

## Initial Setup (Container/Server)

### 1. Clone Repository

```bash
git clone <repository-url>
cd CusVRP
```

### 2. Python Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. OSRM Data Setup (One-time, 10-20 minutes)

```bash
chmod +x scripts/setup_osrm_korea.sh
./scripts/setup_osrm_korea.sh
```

This downloads and processes Korea OSM data into `osrm-data/`. Subsequent runs skip existing files.

### 4. Start Services

**Option A: Docker (Recommended)**

```bash
docker compose up -d
docker compose ps  # Verify both services are "Up (healthy)"
```

**Option B: Native Installation**

If Docker is unavailable, install OSRM and VROOM natively:

```bash
# Install OSRM (see INSTALL_NATIVE.md for full details)
sudo apt-get install -y build-essential cmake libboost-all-dev libbz2-dev libstxxl-dev libxml2-dev libzip-dev liblua5.2-dev libtbb-dev
# Build OSRM from source or use pre-built binary
# Process OSM data: osrm-extract, osrm-contract

# Install VROOM (CLI or HTTP server)
# Build from source or use pre-built binary

# Start services
./start_services.sh
```

**Verify Services:**

```bash
# Test OSRM
curl "http://localhost:5001/route/v1/driving/127.385,36.35;127.386,36.36"

# Test VROOM
curl -X POST http://localhost:3000/ \
  -H "Content-Type: application/json" \
  -d '{"vehicles":[{"id":1,"start":[127.385,36.35],"end":[127.385,36.35],"capacity":[100]}],"jobs":[{"id":1,"location":[127.386,36.36],"amount":[10]}]}'
```

---

## Running Experiments

### Basic Usage

```bash
python -m src.vrp_fairness.run_experiment \
  --seed 0 \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 50 \
  --num-dcs 3 \
  --demand-field A26 \
  --eps 0.10 \
  --iters 200 \
  --method proposed \
  --map
```

### Key Arguments

- `--seed N`: Random seed for reproducibility
- `--gpkg PATH`: Load stops from GeoPackage file
- `--sample-n N`: Number of stops to sample (optional, uses all if omitted)
- `--num-dcs N`: Number of distribution centers (DCs) to generate randomly
- `--dcs "lat,lon" ...`: Explicit DC coordinates (alternative to `--num-dcs`)
- `--demand-field FIELD`: Column name for demand/households (e.g., "A26")
- `--layer NAME`: GPKG layer name (default: "yuseong_housing_2__point")
- `--eps FLOAT`: Cost budget tolerance (default: 0.10 = 10% increase allowed)
- `--iters N`: Number of ALNS iterations (default: 200)
- `--method {local|proposed}`: Algorithm method (default: "local")
  - `local`: Local search with time matrix (slower, includes timetable plots)
  - `proposed`: ALNS algorithm (faster, no time matrix needed)
- `--alpha FLOAT`: Weight for Z1 (min-max waiting) in combined objective (default: 0.4)
- `--beta FLOAT`: Weight for Z2 (routing cost) in combined objective (default: 0.3)
- `--gamma FLOAT`: Weight for Z3 (variance) in combined objective (default: 0.3)
- `--map`: Generate interactive HTML map
- `--map-tiles NAME`: Map tile provider (default: "CartoDB Positron")

### Example: Random Stops

```bash
python -m src.vrp_fairness.run_experiment \
  --seed 0 \
  --n 30 \
  --city daejeon \
  --dcs "36.35,127.385" \
  --eps 0.10 \
  --iters 300
```

### Example: With Housing Type Filter

```bash
python -m src.vrp_fairness.run_experiment \
  --seed 0 \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 30 \
  --housing-type 공동주택 \
  --dcs "36.35,127.385" \
  --eps 0.10 \
  --iters 300
```

---

## Output Files

All outputs are saved in `outputs/`:

- `baseline.json` - Baseline VROOM solution with routes
- `baseline_metrics.csv` - Baseline metrics (waiting times, costs)
- `improved.json` - Improved solution (if improvement algorithm ran)
- `comparison.csv` - Baseline vs improved comparison
- `solutions/seed{N}_n{M}_{city}_{method}.json` - Full solution JSON
- `traces/seed{N}_n{M}_{city}_{method}.csv` - ALNS iteration trace
- `plots/seed{N}_n{M}_{city}_routes.png` - Route visualization
- `plots/seed{N}_n{M}_{city}_wait_hist.png` - Waiting time histogram (if time_matrix available)
- `maps/seed{N}_n{M}_{city}_routes.html` - Interactive Folium map (if `--map` used)
- `cache_polyline.json` - Cached OSRM polyline responses

---

## Interactive Map Visualization

Maps show baseline and improved routes with exclusive toggling (radio-button behavior).

**Viewing Maps:**

1. **Direct file open**: Double-click HTML file (map tiles may not load offline)
2. **HTTP server**:
   ```bash
   cd outputs/maps
   python3 -m http.server 8000
   # Access: http://localhost:8000/seed0_n50_daejeon_routes.html
   ```

**Map Features:**

- **Stops**: Blue circle markers (always visible)
- **Depots**: Red warehouse icons (always visible)
- **Baseline Routes**: Solid colored lines (default: ON)
- **Improved Routes**: Dashed colored lines (default: OFF)
- **Layer Control**: Toggle baseline/improved routes exclusively
- **OSRM Geometry**: Routes follow actual road network (curvy polylines)

---

## Project Structure

```
src/vrp_fairness/
├── assignment.py          # Stop-to-depot assignment using OSRM
├── config.py              # Experiment configuration
├── data.py                # Stop generation/loading (GPKG support)
├── geometry.py            # Geometry parsing (polyline, GeoJSON)
├── inavi.py               # OSRM/iNavi caching (polyline cache)
├── local_search.py        # Local search improvement algorithm
├── map_folium.py          # Interactive map generation
├── metrics.py             # Solution metrics calculation
├── objectives.py          # Objective functions (Z1, Z2, Z3, combined Z)
├── osrm_provider.py       # OSRM time/distance providers
├── plotting.py            # Static route plots
├── proposed_algorithm.py  # ALNS fairness optimization
├── run_experiment.py      # Main CLI entry point
└── vroom_vrp.py           # VROOM VRP solver wrapper

scripts/
├── clean_geometry_from_json.py  # Utility: remove geometry from JSON
├── generate_map_from_json.py    # Utility: generate map from existing JSON
├── preview_gpkg.py             # Preview GPKG file contents
├── setup_osrm_korea.sh          # OSRM data setup script
└── test_proposed_on_realdata.py # Smoke test for proposed algorithm
```

---

## Environment Variables

- `VROOM_BASE_URL`: VROOM API URL (default: `http://localhost:3000/`)
- `OSRM_ADDRESS`: OSRM routing service URL (default: `http://localhost:5001`)
- `MCP_SERVER_COMMAND`: MCP server path for iNavi API (optional)
- `INAVI_APPKEY`: iNavi API key (optional, for real routing)

---

## Troubleshooting

### OSRM Container Fails to Start

**Check**: Does `osrm-data/korea-latest.osrm.fileIndex` exist?

```bash
ls -lh osrm-data/korea-latest.osrm*
```

If missing, run setup script:
```bash
./scripts/setup_osrm_korea.sh
```

### VROOM Not Responding

1. **Check OSRM is healthy**:
   ```bash
   docker compose ps
   curl "http://localhost:5001/route/v1/driving/127.0,36.0;127.1,36.1"
   ```

2. **Check VROOM logs**:
   ```bash
   docker compose logs vroom
   # Or for native: cat vroom.log
   ```

3. **Verify port availability**:
   ```bash
   lsof -i :3000
   lsof -i :5001
   ```

### Python Import Errors

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Map Tiles Not Loading

- Map tiles load from external CDNs (OpenStreetMap/CartoDB)
- If offline, routes will still be visible but map background may be blank
- Try different tile provider: `--map-tiles "OpenStreetMap"`

### Time Matrix Missing (for `--method local`)

The `local` method requires a full time matrix. If you see `KeyError: 'time_matrix'`:

- Use `--method proposed` instead (doesn't require time matrix)
- Or ensure time matrix is built (automatic for `local` method)

### Services Already Running

```bash
# Stop Docker services
docker compose down

# Stop native services
pkill -f "osrm-routed"
pkill -f "vroom"
```

---

## Algorithm Details

### Baseline Solution

1. **Stop Assignment**: Each stop assigned to nearest depot using OSRM travel time
2. **Per-Depot VRP**: Independent VRP solved for each depot using VROOM
3. **Geometry**: OSRM route geometry included in solution

### Proposed Algorithm (ALNS)

Adaptive Large Neighborhood Search optimizes:

```
Z = α·(Z1/Z1*) + β·(Z2/Z2*) + γ·(Z3/Z3*)
```

Where:
- **Z1**: `max_i (n_i · w_i)` - Maximum weighted waiting time
- **Z2**: `Σ(road distances)` or `Σ(travel times)` - Total routing cost
- **Z3**: `Σ n_i · (w_i - w̄)²` - Weighted variance of waiting times
- **Z1*, Z2*, Z3***: Baseline values (normalization)

**Constraints:**
- Z2 ≤ (1 + ε) · Z2* (cost budget)
- Vehicle capacity (if `--enforce-capacity`)

**Operators:**
- **Destroy**: Random stop removal (removes geometry to force recalculation)
- **Repair**: Regret-2 insertion with incremental evaluation

### CTS-ALNS: Contextual Thompson Sampling for Operator Selection

The ALNS algorithm can use **Contextual Thompson Sampling (CTS)** to adaptively select destroy/repair operator pairs based on the current solution state.

**Operator Pairs:**
- `worst_k + regret2`: Remove worst stops, regret-2 insertion
- `worst_k + best_insert`: Remove worst stops, best insertion
- `cluster_k + regret2`: Remove clustered stops, regret-2 insertion
- `cluster_k + best_insert`: Remove clustered stops, best insertion
- `random_k + regret2`: Remove random stops, regret-2 insertion
- `random_k + best_insert`: Remove random stops, best insertion

**Context Features:**
- Normalized objectives (Z1/Z1*, Z2/Z2*, Z3/Z3*)
- Cost budget slack
- DC imbalance (route duration variation)
- Tail ratio (top-10% weighted waiting / overall mean)
- Boundary ratio (fraction of stops near multiple depots)
- Iteration progress

**Reward Function:**
- Positive reward for Z1 improvement (if within cost budget)
- Negative penalty for cost budget violations

Use `--operator-mode cts` to enable CTS-based operator selection.

---

## Requirements

- Python 3.8+
- Docker Desktop (for Docker setup) OR native OSRM/VROOM installation
- See `requirements.txt` for Python dependencies

---

## Quick Test

```bash
# Run smoke test
python scripts/test_proposed_on_realdata.py \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --layer yuseong_housing_2__point \
  --n 30 \
  --alpha 0.4 \
  --beta 0.3 \
  --gamma 0.3 \
  --eps 0.1 \
  --iters 100 \
  --seed 0
```

## Example Commands

### Single CTS Run

```bash
python -m src.vrp_fairness.run_experiment \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 60 \
  --eps 0.10 \
  --alpha 1.0 \
  --beta 0.2 \
  --gamma 0.2 \
  --iters 200 \
  --operator-mode cts \
  --method proposed
```

### Comparison Experiment (ALNS vs ALNS+CTS)

```bash
python scripts/compare_cts_vs_alns.py \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --n-stops 100 \
  --eps 0.10 \
  --alpha 1.0 \
  --beta 0.2 \
  --gamma 0.2 \
  --iters 200 \
  --runs 10
```

This will:
- Run 10 experiments with different seeds
- Compare fixed operator selection vs CTS-based selection
- Generate summary statistics and trace files
- Save results to `outputs/experiments/cts_vs_alns_results.csv`

---

## License

[Add license information]
