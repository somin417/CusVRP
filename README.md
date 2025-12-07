# VRP Fairness Improvement System

Python system for solving Vehicle Routing Problems (VRP) with fairness (waiting-time) optimization. Generates baseline routes using VROOM, then applies local search to improve fairness while respecting cost budgets.

## Features

- **Baseline VRP**: 3-vehicle routes per DC using VROOM VRP solver
- **Fairness Improvement**: Local search to minimize maximum waiting time
- **Road Routing**: iNavi API for real distances/times and route polylines
- **Multiple Data Sources**: Random generation or CSV loading
- **Reproducible**: Deterministic seeds and experiment logging

## Quick Start

### 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Docker Services (VROOM + OSRM)

```bash
# One-time: Download and process OSM data (10-20 min)
chmod +x scripts/setup_osrm_korea.sh
./scripts/setup_osrm_korea.sh

# Start services
docker compose up -d
```

See `README_DOCKER.md` for detailed Docker setup.

### 3. Run Experiment

```bash
python -m src.vrp_fairness.run_experiment \
    --seed 0 --n 30 --city daejeon \
    --dcs "36.35,127.385" --eps 0.10 --iters 300
```

## Project Structure

```
src/vrp_fairness/
├── config.py          # Experiment configuration
├── data.py            # Stop generation/loading
├── inavi.py           # iNavi road routing & caching
├── vroom_vrp.py       # VROOM VRP solver wrapper
├── metrics.py         # Waiting time & cost metrics
├── local_search.py    # Fairness improvement algorithm
├── plotting.py        # Route visualization
└── run_experiment.py  # Main CLI entry point
```

## Configuration

### Environment Variables

- `VROOM_BASE_URL`: VROOM API URL (default: `http://localhost:3000/`)
- `MCP_SERVER_COMMAND`: MCP server path for iNavi API (optional)
- `INAVI_APPKEY`: iNavi API key (optional, for real routing)

### City Configuration

Daejeon bounding box and default DC locations are configured in `config.py`.

## Usage Examples

### Approximate Mode (Haversine Distance)

```bash
python -m src.vrp_fairness.run_experiment \
    --seed 0 --n 30 --city daejeon \
    --dcs "36.35,127.385" --eps 0.10 --iters 300 --approx
```

### Baseline Only

```bash
python -m src.vrp_fairness.run_experiment \
    --seed 0 --n 60 --city daejeon \
    --dcs "36.35,127.385" --baseline-only
```

### With CSV Stops

```bash
python -m src.vrp_fairness.run_experiment \
    --seed 0 --city daejeon \
    --dcs "36.35,127.385" \
    --stops-file demand_points_daejeon.csv
```

## Output Files

All outputs are saved in `outputs/`:

- `baseline.json` - Baseline solution with routes
- `baseline_metrics.csv` - Baseline metrics
- `improved.json` - Improved solution
- `comparison.csv` - Baseline vs improved comparison
- `plots/` - Route visualizations and waiting time histograms
- `cache_inavi.json` - Cached iNavi API responses

## Documentation

- `README_DOCKER.md` - Docker setup for VROOM + OSRM
- `MCP_SETUP.md` - iNavi API setup via MCP (optional)

## Requirements

- Python 3.8+
- Docker Desktop (for VROOM + OSRM)
- See `requirements.txt` for Python dependencies
