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

### Core Experiment Files

**JSON Solutions:**
- `baseline.json` - Baseline VROOM solution with routes
- `local.json` - Local search solution
- `improved.json` - ALNS improved solution
- `proposed_debug.json` - ALNS debug information (objectives, trace, normalizers)
- `variance_vs_mad_results.json` - Z3 Variance vs MAD comparison results
- `solutions/seed*_alns.json` - Full solution backups
- `best_solutions/seed*_alns_best.json` - Best solution backups

**CSV Data:**
- `baseline_metrics.csv` - Baseline metrics (waiting times, costs)
- `comparison.csv` - Baseline vs improved comparison
- `compare_scores.csv` - Z1, Z2, Z3, Z scores per method
- `compare_metrics.csv` - Key metrics deltas vs baseline
- `compare_wait_values.csv` - Waiting time values for plotting
- `traces/seed*_proposed.csv` - ALNS iteration trace

**Visualizations:**
- `compare_wait_panels.png` - Waiting time comparison histogram (from `compare_waiting_and_scores.py`)
- `cts_vs_alns_wait_panels.png` - ALNS vs CTS comparison (from `compare_cts_vs_alns.py`)
- `maps/{run_id}_routes.html` - Interactive Folium map (if `--map` used)

**Cache:**
- `cache_polyline.json`, `cache_osrm_polyline.json` - Cached OSRM responses

### Recommended Workflow

**1. Run Basic Comparison Experiment:**
```bash
python scripts/compare_waiting_and_scores.py \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 50 \
  --num-dcs 3 \
  --demand-field A26 \
  --eps 0.10 \
  --iters 200
```
**Generates:** `baseline.json`, `local.json`, `improved.json`, `compare_*.csv`, `compare_wait_panels.png`

**2. Run CTS Comparison (Optional):**
```bash
python scripts/compare_cts_vs_alns.py \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 50 \
  --num-dcs 3 \
  --eps 0.10 \
  --iters 200
```
**Generates:** `cts_vs_alns_*.csv`, `cts_vs_alns_wait_panels.png`  
**Reuses:** `baseline.json`, `improved.json` from step 1

**3. Run Z3 Variance vs MAD Comparison (Optional):**
```bash
python scripts/compare_Z3_variance_vs_MAD.py \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 50 \
  --num-dcs 3 \
  --eps 0.10 \
  --iters 200
```
**Generates:** `variance_vs_mad_*.json`, `compare_wait_values_z3.csv`  
**Reuses:** Can use `--reuse` flag to reuse baseline and MAD solution from step 1

**4. Generate Additional Plots and Maps:**
```bash
# Generate waiting plots and map from JSON files
python scripts/utils/generate_from_json.py \
  --baseline outputs/baseline.json \
  --improved outputs/improved.json \
  --local outputs/local.json

# Generate weighted waiting graph from CSV files
python scripts/utils/generate_weighted_waiting_graph.py
```
**Generates:** `waiting_plot.png`, `weighted_waiting_plot.png`, `map_compare.html`, `weighted_waiting_graph.png`

### Script Summary

| Script | Main Outputs |
|--------|-------------|
| `compare_waiting_and_scores.py` | Baseline/Local/ALNS comparison (CSV, PNG) |
| `compare_cts_vs_alns.py` | ALNS vs CTS comparison (CSV, PNG) |
| `compare_Z3_variance_vs_MAD.py` | Variance vs MAD Z3 comparison (JSON, CSV) |
| `scripts/utils/generate_from_json.py` | Plots and maps from JSON files (PNG, HTML) |
| `scripts/utils/generate_weighted_waiting_graph.py` | Weighted waiting graph from CSV files (PNG) |

**Note:** Plot HTML generation has been removed. Only PNG plots are generated. HTML is only generated for interactive maps.

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
├── compare_waiting_and_scores.py    # Main comparison: Baseline/Local/ALNS
├── compare_cts_vs_alns.py           # ALNS vs CTS comparison
├── compare_Z3_variance_vs_MAD.py   # Z3 Variance vs MAD comparison
├── setup_osrm_korea.sh              # OSRM data setup script
└── utils/
    ├── generate_from_json.py        # Generate plots/maps from JSON files
    ├── generate_weighted_waiting_graph.py  # Generate weighted waiting graph from CSV
    ├── plot_from_json.py            # Plot from baseline/improved JSON
    ├── plotting_utils.py            # Common plotting utilities
    ├── utils.py                     # Common utilities (load_json, etc.)
    ├── clean_geometry_from_json.py  # Remove geometry from JSON
    ├── fix_best_solution_from_trace.py  # Fix best solution from trace
    ├── preview_gpkg.py              # Preview GPKG file contents
    └── validate_*.py                # Validation scripts
```

---

## Environment Variables

- `VROOM_BASE_URL`: VROOM API URL (default: `http://localhost:3000/`)
- `OSRM_ADDRESS`: OSRM routing service URL (default: `http://localhost:5001`)

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

## Validation Framework

The validation framework provides two complementary scripts to validate and compare ALNS (Fixed) and CTS algorithms on single-seed runs.

### Validation 1: Correctness Validation

Validates that both algorithms produce valid, consistent solutions:

```bash
python scripts/validate_alns_cts_correctness.py \
  --baseline outputs/baseline.json \
  --alns outputs/improved.json \
  --cts outputs/cts_solution.json \
  --eps 0.10 \
  --alpha 0.5 \
  --beta 0.3 \
  --gamma 0.2
```

**Checks:**
- All stops are assigned (no missing stops)
- Solution structure is valid (routes_by_dc, ordered_stop_ids, etc.)
- Objectives are computed correctly (Z1, Z2, Z3, Z)
- Budget constraints are respected
- Waiting times are computed for all stops
- Both algorithms use the same baseline, stops, and depots

**Outputs:**
- `outputs/validation_correctness_report.json`: Detailed validation results
- `outputs/validation_correctness_summary.txt`: Human-readable summary

### Validation 2: Comprehensive Comparison

Compares ALNS and CTS results with detailed metrics:

```bash
python scripts/validate_alns_cts_comparison.py \
  --baseline outputs/baseline.json \
  --alns outputs/improved.json \
  --cts outputs/cts_solution.json \
  --alns-debug outputs/proposed_debug.json \
  --cts-debug outputs/cts_debug.json \
  --alpha 0.5 \
  --beta 0.3 \
  --gamma 0.2
```

**Comparison Metrics:**
- Objective values: Z, Z1, Z2, Z3 (absolute and relative to baseline)
- Waiting time distribution: mean, median, max, p95, p99
- Fairness metrics: weighted waiting time variance, top-10% vs overall mean
- Route structure: number of routes, route lengths, depot utilization
- Convergence: iteration where best solution was found

**Outputs:**
- `outputs/validation_comparison_report.json`: Detailed comparison metrics
- `outputs/validation_comparison_summary.txt`: Human-readable summary
- `outputs/validation_comparison_metrics.csv`: Tabular comparison

**Usage Workflow:**

1. Run comparison experiment:
   ```bash
   python scripts/compare_cts_vs_alns.py --sample-n 50 --iters 50
   ```

2. Validate correctness:
   ```bash
   python scripts/validate_alns_cts_correctness.py
   ```

3. Compare results:
   ```bash
   python scripts/validate_alns_cts_comparison.py
   ```

---
