#!/usr/bin/env python3
"""
Remove geometry from solution JSON files so maps recalculate from stop order.
"""

import argparse
import json
import sys
from pathlib import Path


def clean_geometry(json_file: Path, backup: bool = True):
    """Remove geometry and legs from routes in JSON file."""
    print(f"Cleaning geometry from: {json_file}")
    
    if backup:
        backup_file = json_file.with_suffix('.json.backup')
        print(f"Creating backup: {backup_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)
        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    routes_by_dc = data.get("routes_by_dc", {})
    removed_count = 0
    
    for dc_id, routes in routes_by_dc.items():
        for route in routes:
            if "geometry" in route:
                del route["geometry"]
                removed_count += 1
            if "legs" in route:
                del route["legs"]
                removed_count += 1
    
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Removed geometry from {removed_count} route entries")
    print(f"✓ Updated: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Remove geometry from solution JSON")
    parser.add_argument("json_file", type=str, help="JSON file to clean")
    parser.add_argument("--no-backup", action="store_true", help="Don't create backup")
    
    args = parser.parse_args()
    
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)
    
    clean_geometry(json_path, backup=not args.no_backup)


if __name__ == "__main__":
    main()

