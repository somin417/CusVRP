# iNavi API Setup (Optional)

iNavi Maps API provides real road routing for accurate distances, travel times, and route polylines.

## Quick Setup

### 1. Install MCP Server

```bash
pip install tms-mcp
```

### 2. Configure Environment

```bash
export MCP_SERVER_COMMAND="/Users/isomin/HSS437/venv/bin/tms-mcp"
export INAVI_APPKEY="your_appkey"  # If available
```

### 3. Usage

The system automatically uses iNavi API when configured. Falls back to haversine distance if unavailable.

## What It Provides

- **Road Distances**: Real road distances (not straight-line)
- **Travel Times**: Accurate travel times based on road conditions
- **Route Polylines**: Sequences of road points for visualization

## Configuration

iNavi API requires an `appkey`. Get one from iNavi documentation.

**Without API key**: System uses haversine distance (still works!)

## See Also

- `README_DOCKER.md` - VROOM setup (required for VRP solving)
- `README.md` - Main documentation
