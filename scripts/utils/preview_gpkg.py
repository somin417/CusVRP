#!/usr/bin/env python3
"""
Preview GeoPackage file contents.
"""

import sys
import sqlite3
import pandas as pd
from pathlib import Path

def preview_gpkg(gpkg_path: str, layer: str = "yuseong_housing_2__point"):
    """Print GPKG file information."""
    if not Path(gpkg_path).exists():
        print(f"Error: File not found: {gpkg_path}")
        sys.exit(1)
    
    conn = sqlite3.connect(gpkg_path)
    
    # Get table info
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{layer}'")
    if not cursor.fetchone():
        print(f"Error: Layer '{layer}' not found in {gpkg_path}")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cursor.fetchall()]
        print(f"Available tables: {tables}")
        conn.close()
        sys.exit(1)
    
    # Get row count
    cursor.execute(f"SELECT COUNT(*) FROM {layer}")
    row_count = cursor.fetchone()[0]
    
    # Get column names
    cursor.execute(f"PRAGMA table_info({layer})")
    columns = [col[1] for col in cursor.fetchall()]
    
    # Load sample rows
    df = pd.read_sql_query(f"SELECT * FROM {layer} LIMIT 5", conn)
    
    # Get bounding box
    if 'latitude' in columns and 'longitude' in columns:
        bbox_df = pd.read_sql_query(
            f"SELECT MIN(latitude) as lat_min, MAX(latitude) as lat_max, "
            f"MIN(longitude) as lon_min, MAX(longitude) as lon_max FROM {layer}",
            conn
        )
        bbox = bbox_df.iloc[0]
    else:
        bbox = None
    
    conn.close()
    
    # Print results
    print("=" * 80)
    print(f"GeoPackage: {gpkg_path}")
    print(f"Layer: {layer}")
    print("=" * 80)
    print(f"Row count: {row_count:,}")
    print(f"Columns: {', '.join(columns)}")
    print()
    
    if bbox is not None:
        print("Bounding box:")
        print(f"  Latitude:  {bbox['lat_min']:.6f} to {bbox['lat_max']:.6f}")
        print(f"  Longitude: {bbox['lon_min']:.6f} to {bbox['lon_max']:.6f}")
        print()
    
    print("Sample rows (first 5):")
    print(df.to_string())
    print()
    
    # Check for housing type distribution
    if 'A9' in columns:
        conn2 = sqlite3.connect(gpkg_path)
        type_df = pd.read_sql_query(f"SELECT A9, COUNT(*) as count FROM {layer} GROUP BY A9", conn2)
        conn2.close()
        print("Housing type distribution:")
        print(type_df.to_string())
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_gpkg.py <gpkg_path> [layer]")
        sys.exit(1)
    
    gpkg_path = sys.argv[1]
    layer = sys.argv[2] if len(sys.argv) > 2 else "yuseong_housing_2__point"
    
    preview_gpkg(gpkg_path, layer)

