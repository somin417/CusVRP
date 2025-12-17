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

## Quick Start: Reproducing Experiments

This section provides exact commands to reproduce all experiments with consistent results.

### Prerequisites

1. **Services Running**: Ensure OSRM and VROOM services are running (see "Start Services" above)
2. **Data Files**: Ensure `data/yuseong_housing_3__point.gpkg` exists
3. **Python Environment**: Activate virtual environment and install dependencies

### Experiment Execution Order

Experiments should be run in the following order due to dependencies:

1. **Baseline, Local, and ALNS (MAD)**: `compare_waiting_and_scores.py`
2. **ALNS (Variance) vs MAD**: `compare_Z3_variance_vs_MAD.py` (can reuse step 1 results)
3. **CTS vs ALNS**: `compare_cts_vs_alns.py` (reuses step 1 results)
4. **ABC Balance Models**: `run_cts_abc_balance.py` (requires baseline from step 1)

### Experiment 1: Baseline, Local, and ALNS (MAD)

**Purpose**: Generate baseline, local search, and ALNS (MAD) solutions.

**Command**:
```bash
python scripts/compare_waiting_and_scores.py
```

**Default Parameters**:
- `--seed 42`: Random seed for reproducibility
- `--sample-n 50`: Number of stops to sample
- `--num-dcs 3`: Number of distribution centers
- `--iters 50`: ALNS iterations
- `--eps 0.10`: Cost budget tolerance (10% increase allowed)
- `--alpha 0.5`, `--beta 0.3`, `--gamma 0.2`: Objective weights

**Outputs**:
- `outputs/solutions/baseline.json`
- `outputs/solutions/local.json`
- `outputs/solutions/ALNS_MAD.json`
- `outputs/plots/compare_wait_panels.png`
- `outputs/data/baseline_local_alns_mad_*.csv`

### Experiment 2: ALNS (Variance) vs MAD

**Purpose**: Compare ALNS using Variance vs MAD for Z3.

**Command** (with reuse):
```bash
python scripts/compare_Z3_variance_vs_MAD.py --reuse
```

**Command** (full run):
```bash
python scripts/compare_Z3_variance_vs_MAD.py
```

**Default Parameters**:
- `--seed 42`: Random seed
- `--sample-n 50`: Number of stops
- `--num-dcs 3`: Number of DCs
- `--iters 50`: ALNS iterations
- `--eps 0.10`: Cost budget tolerance
- `--alpha 0.5`, `--beta 0.3`, `--gamma 0.2`: Objective weights

**Outputs**:
- `outputs/solutions/ALNS_VAR.json` (only if not using `--reuse`)
- `outputs/data/baseline_alns_variance_vs_mad_*.csv`

### Experiment 3: CTS vs ALNS

**Purpose**: Compare ALNS with fixed operators vs Contextual Thompson Sampling.

**Command**:
```bash
python scripts/compare_cts_vs_alns.py
```

**Default Parameters**:
- `--seed 42`: Random seed
- `--sample-n 50`: Number of stops
- `--num-dcs 3`: Number of DCs
- `--iters 50`: ALNS iterations
- `--eps 0.30`: Cost budget tolerance (30% increase allowed, higher than other experiments for more exploration)
- `--alpha 0.5`, `--beta 0.3`, `--gamma 0.2`: Objective weights

**Note**: This script reuses results from Experiment 1. Use `--force-rerun-alns` to regenerate ALNS results.

**Outputs**:
- `outputs/solutions/cts_solution.json`
- `outputs/plots/cts_vs_alns_wait_panels.png`
- `outputs/data/cts_vs_alns_*.csv`

### Experiment 4: ABC Balance Models

**Purpose**: Test different alpha/beta/gamma weight combinations.

**Command**:
```bash
python scripts/run_cts_abc_balance.py
```

**Default Parameters**:
- `--seed 42`: Random seed
- `--sample-n 20`: Number of stops to sample
- `--iters 20`: CTS iterations per configuration
- `--eps 0.30`: Cost budget tolerance (30% increase allowed)
- Configurations tested:
  - Z1-focused: alpha=0.6, beta=0.3, gamma=0.1
  - Balanced: alpha=0.35, beta=0.3, gamma=0.35
  - Z3-focused: alpha=0.1, beta=0.3, gamma=0.6

**Prerequisites**: Requires `outputs/solutions/baseline.json` from Experiment 1.

**Outputs**:
- `outputs/solutions/abc_balance/cts_*_best.json` (3 files)
- `outputs/plots/abc_balance_comparison.png`
- `outputs/data/abc_balance_summary.json`

### Generating Additional Plots

After running experiments, generate additional comparison plots:

**Baseline, ALNS, CTS Comparison**:
```bash
python scripts/generate_waiting_plots_baseline_alns_cts.py
```
Outputs: `waiting_plot.png`, `weighted_waiting_plot.png`

**ABC Balance Models Comparison**:
```bash
python scripts/generate_waiting_times_z.py
python scripts/generate_weighted_waiting_z.py
```
Outputs: `waiting_times_z.png`, `weighted_waiting_times_z.png`

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

All outputs are saved in `outputs/` and organized by type:

### Directory Structure

```
outputs/
├── solutions/          # All JSON solution files
│   ├── baseline.json
│   ├── local.json          # Local search algorithm (different from ALNS)
│   ├── ALNS_MAD.json       # ALNS solution using MAD for Z3
│   ├── ALNS_VAR.json       # ALNS solution using Variance for Z3
│   └── cts_solution.json   # ALNS with CTS operator selection
├── debug/              # All debug JSON files
│   ├── proposed_debug.json
│   ├── cts_debug.json
│   └── variance_vs_mad_*.json
├── plots/              # All PNG visualization files
│   ├── compare_wait_panels.png
│   ├── waiting_times_baseline_alns_cts.png
│   ├── weighted_waiting_times_baseline_alns_cts.png
│   ├── waiting_times_abc_balance_models.png
│   ├── weighted_waiting_times_abc_balance_models.png
│   ├── abc_balance_comparison.png
│   └── cts_vs_alns_wait_panels.png
├── data/               # All CSV data files
│   ├── baseline_metrics.csv
│   ├── baseline_vs_local_comparison.csv
│   ├── baseline_vs_alns_mad_comparison.csv
│   ├── baseline_vs_cts_comparison.csv
│   ├── baseline_local_alns_mad_scores.csv
│   ├── baseline_local_alns_mad_metrics.csv
│   ├── baseline_local_alns_mad_wait_values.csv
│   ├── baseline_local_alns_mad_wait_hist_data.csv
│   ├── baseline_alns_variance_vs_mad_scores.csv
│   ├── baseline_alns_variance_vs_mad_wait_values.csv
│   ├── cts_vs_alns_scores.csv
│   ├── cts_vs_alns_wait_values.csv
│   ├── cts_vs_alns_wait_hist_data.csv
│   ├── abc_balance_summary.json
│   └── ...
├── maps/               # HTML map files
│   └── {run_id}_routes.html
├── cache/              # Cache files
│   ├── cache_polyline.json
│   └── cache_osrm_polyline.json
├── best_solutions/     # Best solution backups
│   ├── abc_balance/
│   └── seed*_*_best.json
├── solutions/          # Full solution backups (subdirectory)
│   └── seed*_*.json
└── traces/             # Iteration trace CSV files
    └── seed*_*.csv
```

### Core Experiment Files

**JSON Solutions (`outputs/solutions/`):**
- `baseline.json` - Baseline VROOM solution with routes (generated by all experiments)
- `local.json` - Local search solution (from `compare_waiting_and_scores.py`)
- `ALNS_MAD.json` - ALNS improved solution using MAD for Z3 (from `compare_waiting_and_scores.py`)
- `ALNS_VAR.json` - ALNS improved solution using Variance for Z3 (from `compare_Z3_variance_vs_MAD.py`)
- `cts_solution.json` - CTS solution (from `compare_cts_vs_alns.py`)
- `solutions/abc_balance/cts_z1_focused_best.json` - Z1-focused ABC balance solution
- `solutions/abc_balance/cts_balanced_best.json` - Balanced ABC balance solution
- `solutions/abc_balance/cts_z3_focused_best.json` - Z3-focused ABC balance solution

**Debug Files (`outputs/debug/`):**
- `alns_mad_debug.json` - ALNS (MAD) debug information (objectives, trace, normalizers)
- `cts_debug.json` - CTS debug information
- `variance_vs_mad_MAD_debug.json` - MAD experiment debug info
- `variance_vs_mad_VARIANCE_debug.json` - Variance experiment debug info
- `trace_analysis.json` - Trace analysis results

**CSV Data (`outputs/data/`):**
- `baseline_metrics.csv` - Baseline metrics (waiting times, costs)
- `baseline_vs_local_comparison.csv` - Baseline vs local search comparison
- `baseline_vs_alns_mad_comparison.csv` - Baseline vs ALNS (MAD) comparison
- `baseline_vs_cts_comparison.csv` - Baseline vs CTS comparison
- `baseline_local_alns_mad_scores.csv` - Z1, Z2, Z3, Z scores per method (from `compare_waiting_and_scores.py`)
- `baseline_local_alns_mad_metrics.csv` - Key metrics deltas vs baseline
- `baseline_local_alns_mad_wait_values.csv` - Waiting time values for plotting
- `baseline_local_alns_mad_wait_hist_data.csv` - Histogram data for plotting
- `baseline_alns_variance_vs_mad_scores.csv` - Z3 variance vs MAD comparison scores
- `baseline_alns_variance_vs_mad_wait_values.csv` - Z3 comparison waiting time values
- `cts_vs_alns_scores.csv` - ALNS vs CTS comparison scores
- `cts_vs_alns_wait_values.csv` - ALNS vs CTS waiting time values
- `abc_balance_summary.json` - ABC balance experiment summary
- `variance_vs_mad_results.json` - Z3 Variance vs MAD comparison results
- `weighted_waiting_graph_hist_data.csv` - Weighted waiting histogram data
- `weighted_waiting_graph_values.csv` - Weighted waiting time values
- `traces/seed*_alns.csv` - ALNS iteration trace
- `traces/seed*_cts.csv` - CTS iteration trace

**Visualizations (`outputs/plots/`):**
- `compare_wait_panels.png` - Waiting time comparison histogram (Baseline/Local/ALNS from `compare_waiting_and_scores.py`)
- `waiting_plot.png` - Waiting time distribution (Baseline/ALNS/CTS from `generate_waiting_plots_baseline_alns_cts.py`)
- `weighted_waiting_plot.png` - Weighted waiting time distribution (Baseline/ALNS/CTS)
- `waiting_times_z.png` - Waiting time distribution for ABC balance models (from `generate_waiting_times_z.py`)
- `weighted_waiting_times_z.png` - Weighted waiting time distribution for ABC balance models (from `generate_weighted_waiting_z.py`)
- `abc_balance_comparison.png` - Z scores comparison across ABC balance configurations (from `run_cts_abc_balance.py`)
- `cts_vs_alns_wait_panels.png` - ALNS vs CTS comparison (from `compare_cts_vs_alns.py`)
- `weighted_waiting_graph.png` - Combined weighted waiting graph (from `generate_weighted_waiting_graph.py`)

**Maps (`outputs/maps/`):**
- `map_compare.html` - Interactive comparison map
- `{run_id}_routes.html` - Interactive route map (if `--map` used)

**Cache (`outputs/cache/`):**
- `cache_polyline.json`, `cache_osrm_polyline.json` - Cached OSRM responses
- `cache_inavi.json` - iNavi cache

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
**Generates:** 
- `solutions/baseline.json`, `solutions/local.json`, `solutions/improved.json`
- `data/compare_*.csv`
- `plots/compare_wait_panels.png`
- `debug/proposed_debug.json`

**2. Run CTS Comparison (Optional):**
```bash
python scripts/compare_cts_vs_alns.py \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 50 \
  --num-dcs 3 \
  --eps 0.10 \
  --iters 200
```
**Generates:** 
- `solutions/cts_solution.json`
- `data/cts_vs_alns_*.csv`
- `plots/cts_vs_alns_wait_panels.png`
- `debug/cts_debug.json`
**Reuses:** `solutions/baseline.json`, `solutions/improved.json` from step 1

**3. Run Z3 Variance vs MAD Comparison (Optional):**
```bash
python scripts/compare_Z3_variance_vs_MAD.py \
  --gpkg data/yuseong_housing_3__point.gpkg \
  --sample-n 50 \
  --num-dcs 3 \
  --eps 0.10 \
  --iters 200
```
**Generates:** 
- `solutions/variance_solution.json`
- `data/variance_vs_mad_results.json`
- `data/compare_wait_values_z3.csv`
- `debug/variance_vs_mad_*.json`
**Reuses:** Can use `--reuse` flag to reuse `solutions/baseline.json` and `solutions/improved.json` from step 1

**4. Run ABC Balance Experiment (Optional):**
```bash
python scripts/run_cts_abc_balance.py \
  --baseline outputs/baseline.json \
  --iters 20 \
  --seed 42 \
  --eps 0.30
```
**Generates:** 
- `best_solutions/abc_balance/cts_z1_focused_best.json` (α=0.6, β=0.3, γ=0.1)
- `best_solutions/abc_balance/cts_balanced_best.json` (α=0.35, β=0.3, γ=0.35)
- `best_solutions/abc_balance/cts_z3_focused_best.json` (α=0.1, β=0.3, γ=0.6)
- `plots/abc_balance_comparison.png` - Z scores comparison plot
- `data/abc_balance_summary.json` - Summary of all configurations

**5. Generate Additional Plots from Existing Solutions:**
```bash
# Generate waiting plots for Baseline/ALNS/CTS
python scripts/generate_waiting_plots_baseline_alns_cts.py

# Generate waiting plots for ABC balance models (requires baseline + abc_balance solutions)
python scripts/generate_waiting_times_z.py
python scripts/generate_weighted_waiting_z.py

# Generate plots and map from JSON files (general utility)
python scripts/utils/generate_from_json.py \
  --baseline outputs/baseline.json \
  --improved outputs/improved.json \
  --local outputs/local.json

# Generate weighted waiting graph from CSV files
python scripts/utils/generate_weighted_waiting_graph.py
```
**Generates:** 
- `plots/waiting_times_baseline_alns_cts.png`, `plots/weighted_waiting_times_baseline_alns_cts.png`
- `plots/waiting_times_abc_balance_models.png`, `plots/weighted_waiting_times_abc_balance_models.png`
- `maps/map_compare.html`, `plots/weighted_waiting_graph.png`

### Script Summary

| Script | Purpose | Main Outputs |
|--------|---------|-------------|
| `compare_waiting_and_scores.py` | Main comparison experiment | Baseline/Local/ALNS solutions, CSV scores, `compare_wait_panels.png` |
| `compare_cts_vs_alns.py` | ALNS vs CTS comparison | CTS solution, `cts_vs_alns_*.csv`, `cts_vs_alns_wait_panels.png` |
| `compare_Z3_variance_vs_MAD.py` | Z3 Variance vs MAD comparison | Variance solution, `variance_vs_mad_*.json`, `compare_wait_values_z3.csv` |
| `run_cts_abc_balance.py` | ABC balance experiment | Three ABC balance solutions, `abc_balance_comparison.png`, `abc_balance_summary.json` |
| `generate_waiting_plots_baseline_alns_cts.py` | Plot generation from solutions | `waiting_times_baseline_alns_cts.png`, `weighted_waiting_times_baseline_alns_cts.png` |
| `generate_waiting_times_z.py` | Plot ABC balance waiting times | `waiting_times_abc_balance_models.png` |
| `generate_weighted_waiting_z.py` | Plot ABC balance weighted waiting | `weighted_waiting_times_abc_balance_models.png` |
| `scripts/utils/generate_from_json.py` | General plot/map utility | Plots and maps from any JSON solutions (PNG, HTML) |
| `scripts/utils/generate_weighted_waiting_graph.py` | Weighted waiting graph | `weighted_waiting_graph.png` from CSV files |

**Note:** Plot HTML generation has been removed. Only PNG plots are generated. HTML is only generated for interactive maps.

### File Naming Convention

**Plot Files:**
- `*_baseline_alns_cts.png` - Comparison plots with Baseline, ALNS, and CTS methods
- `*_abc_balance_models.png` - Plots showing ABC balance model results (z1_focused, balanced, z3_focused)
- `compare_*.png` - Comparison plots from main experiment scripts
- `abc_balance_comparison.png` - Z scores comparison across ABC balance configurations

**Solution Files:**
- `baseline.json` - Always the baseline VROOM solution
- `improved.json` - ALNS solution (from `compare_waiting_and_scores.py`)
- `cts_solution.json` - CTS solution (from `compare_cts_vs_alns.py`)
- `best_solutions/abc_balance/*.json` - ABC balance experiment best solutions

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
├── run_cts_abc_balance.py           # ABC balance experiment (Z1/Z2/Z3 weight variations)
├── generate_waiting_plots_baseline_alns_cts.py  # Generate Baseline/ALNS/CTS plots
├── generate_waiting_times_z.py      # Generate ABC balance waiting time plots
├── generate_weighted_waiting_z.py   # Generate ABC balance weighted waiting plots
├── setup_osrm_korea.sh              # OSRM data setup script
└── utils/
    ├── generate_from_json.py        # Generate plots/maps from JSON files (general utility)
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

## Quick Start for Verification

To verify the code works without running full experiments:

### 1. Verify Services
```bash
# Test OSRM
curl "http://localhost:5001/route/v1/driving/127.385,36.35;127.386,36.36"

# Test VROOM
curl -X POST http://localhost:3000/ \
  -H "Content-Type: application/json" \
  -d '{"vehicles":[{"id":1,"start":[127.385,36.35],"end":[127.385,36.35],"capacity":[100]}],"jobs":[{"id":1,"location":[127.386,36.36],"amount":[10]}]}'
```

### 2. Verify Script Imports
```bash
# Test that all scripts can be imported (without running experiments)
python -c "import sys; sys.path.insert(0, '.'); from scripts.compare_waiting_and_scores import *; print('✓ compare_waiting_and_scores.py')"
python -c "import sys; sys.path.insert(0, '.'); from scripts.compare_cts_vs_alns import *; print('✓ compare_cts_vs_alns.py')"
python -c "import sys; sys.path.insert(0, '.'); from scripts.run_cts_abc_balance import *; print('✓ run_cts_abc_balance.py')"
```

### 3. Regenerate Plots from Existing Solutions
If solutions already exist in `outputs/`, you can regenerate plots:
```bash
# Regenerate Baseline/ALNS/CTS plots
python scripts/generate_waiting_plots_baseline_alns_cts.py

# Regenerate ABC balance plots (if abc_balance solutions exist)
python scripts/generate_waiting_times_z.py
python scripts/generate_weighted_waiting_z.py
```

### 4. Understanding Existing Outputs

**Key Files to Check:**
- `outputs/solutions/baseline.json` - Should exist if any experiment was run
- `outputs/solutions/improved.json` - ALNS solution (if `compare_waiting_and_scores.py` was run)
- `outputs/solutions/cts_solution.json` - CTS solution (if `compare_cts_vs_alns.py` was run)
- `outputs/best_solutions/abc_balance/*.json` - ABC balance solutions (if `run_cts_abc_balance.py` was run)
- `outputs/plots/abc_balance_comparison.png` - Visual comparison of ABC balance results
- `outputs/plots/compare_wait_panels.png` - Main comparison plot

**CSV Files for Analysis:**
- `outputs/data/compare_scores.csv` - Z1, Z2, Z3, Z scores for all methods
- `outputs/data/compare_metrics.csv` - Key metrics deltas vs baseline
- `outputs/data/abc_balance_summary.json` - Summary of ABC balance experiment results

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
