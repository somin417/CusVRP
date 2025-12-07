#!/usr/bin/env python3
"""Quick test of VROOM integration with the algorithm."""

import sys
import os
sys.path.insert(0, '.')

from src.vrp_fairness.vroom_vrp import call_vrp

# Test data
depot = {
    "name": "DC1",
    "coordinates": [36.35, 127.385],  # [lat, lon]
    "demand": 0
}

stops = [
    {"name": "S1", "coordinates": [36.36, 127.386], "demand": 10},
    {"name": "S2", "coordinates": [36.37, 127.387], "demand": 15},
    {"name": "S3", "coordinates": [36.38, 127.388], "demand": 20}
]

vehicles = [
    {"capacity": 100},
    {"capacity": 100},
    {"capacity": 100}
]

print("=" * 60)
print("VROOM VRP Integration Test")
print("=" * 60)
print(f"\nDepot: {depot['name']} at {depot['coordinates']}")
print(f"Stops: {len(stops)}")
print(f"Vehicles: {len(vehicles)}")
print(f"\nVROOM_BASE_URL: {os.getenv('VROOM_BASE_URL', 'http://localhost:3000/')}")

print("\nCalling VROOM API...")
try:
    result = call_vrp(
        depot=depot,
        stops=stops,
        vehicles=vehicles
    )
    
    print("\n✓ SUCCESS!")
    print(f"\nRoutes generated: {len(result.get('routes', []))}")
    print(f"Total duration: {result.get('metrics', {}).get('total_duration', 0)}s")
    print(f"Total distance: {result.get('metrics', {}).get('total_distance', 0):.1f}m")
    
    print("\nRoute details:")
    for route in result.get('routes', []):
        stops_str = " -> ".join(route.get('ordered_stop_ids', []))
        print(f"  {route['vehicle_id']}: {stops_str}")
        print(f"    Duration: {route.get('total_duration', 0)}s, Distance: {route.get('total_distance', 0):.1f}m")
    
    unassigned = result.get('metrics', {}).get('unassigned_stops', [])
    if unassigned:
        print(f"\nUnassigned stops: {unassigned}")
    else:
        print("\n✓ All stops assigned!")
        
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)

