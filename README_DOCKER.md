# Docker Setup for VROOM VRP Service

## Prerequisites

- Docker Desktop installed and running on macOS
- Internet connection (only needed for initial OSM data download)

## Quick Start (macOS)

### Step 1: Setup OSRM Data (One-time setup)

Run the setup script to download and process OSM data:

```bash
chmod +x scripts/setup_osrm_korea.sh
./scripts/setup_osrm_korea.sh
```

**Note**: 
- First run: 10-20 minutes (download + extract + contract)
- Subsequent runs: Skips steps if files already exist (idempotent)
- Processed data saved in `./osrm-data/`

### Step 2: Start Services

```bash
docker compose up -d
```

### Step 3: Check Status

```bash
docker compose ps
```

You should see both services with status "Up (healthy)".

### Step 4: Test VROOM

Once services are running, test with:

```bash
curl -X POST http://localhost:3000/ \
  -H "Content-Type: application/json" \
  -d '{"vehicles":[{"id":1,"start":[127.385,36.35],"end":[127.385,36.35],"capacity":[100]}],"jobs":[{"id":1,"location":[127.386,36.36],"amount":[10]}]}'
```

You should see a JSON response with routes.

## Stop Services

```bash
docker compose down
```

## Services

- **OSRM**: Road routing service on port `5001` (internal: `5000`)
- **VROOM**: VRP solver on port `3000`

## Troubleshooting

### OSRM container fails to start

**Check**: Does `./osrm-data/korea-latest.osrm` exist?

```bash
ls -lh osrm-data/korea-latest.osrm
```

If not, run the setup script:
```bash
./scripts/setup_osrm_korea.sh
```

### VROOM container fails to start

**Check**: Is OSRM healthy?

```bash
docker compose ps
docker compose logs osrm
```

VROOM depends on OSRM, so OSRM must be healthy first.

### Port conflicts

If port 3000 or 5001 is already in use:

```bash
# Check what's using the port
lsof -i :3000
lsof -i :5001

# Stop conflicting services or change ports in docker-compose.yml
```

## Re-downloading OSM Data

To force re-download and reprocess:

```bash
rm -rf osrm-data/*
./scripts/setup_osrm_korea.sh
```
