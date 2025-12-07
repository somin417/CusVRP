# VROOM VRP MCP Server

MCP server that exposes a `vroom_vrp` tool for solving Vehicle Routing Problems using VROOM.

## Installation

```bash
cd mcp-vroom
npm install
npm run build
```

## Start Server

```bash
npm start
```

Or for development:
```bash
npm run dev
```

## Configuration

Set the `VROOM_BASE_URL` environment variable (defaults to `http://localhost:3000/`):

```bash
export VROOM_BASE_URL="http://localhost:3000/"
```

## Cursor Configuration

Add to your Cursor MCP settings (or `mcp.json`):

```json
{
  "mcpServers": {
    "vroom-vrp": {
      "command": "node",
      "args": ["dist/index.js"],
      "cwd": "/Users/isomin/HSS437/mcp-vroom",
      "env": {
        "VROOM_BASE_URL": "http://localhost:3000/"
      }
    }
  }
}
```

## Tool: `vroom_vrp`

### Input Example

```json
{
  "depot": {
    "id": "DC1",
    "lat": 36.35,
    "lon": 127.385
  },
  "vehicles": [
    {"id": "V1", "depot_id": "DC1", "capacity": 100},
    {"id": "V2", "depot_id": "DC1", "capacity": 100},
    {"id": "V3", "depot_id": "DC1", "capacity": 100}
  ],
  "stops": [
    {"id": "S1", "lat": 36.36, "lon": 127.386, "demand": 10, "service_time_s": 300},
    {"id": "S2", "lat": 36.37, "lon": 127.387, "demand": 15, "service_time_s": 300}
  ]
}
```

### Output Format

```json
{
  "routes": [
    {
      "vehicle_id": "V1",
      "total_duration_s": 3600,
      "total_distance_m": 5000,
      "steps": [
        {"stop_id": null, "lat": 36.35, "lon": 127.385, "arrival_s": 0, "distance_m": 0},
        {"stop_id": "S1", "lat": 36.36, "lon": 127.386, "arrival_s": 600, "distance_m": 2000},
        {"stop_id": null, "lat": 36.35, "lon": 127.385, "arrival_s": 3600, "distance_m": 5000}
      ]
    }
  ],
  "summary": {
    "total_duration_s": 3600,
    "total_distance_m": 5000,
    "unassigned_stops": []
  }
}
```

## Requirements

- Node.js 18+
- VROOM server running (see `../README_DOCKER.md`)
- Docker Desktop (for VROOM + OSRM)

