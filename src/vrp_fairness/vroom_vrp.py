"""
VROOM VRP solver wrapper and response parser.
"""

import json
import logging
import os
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def call_vrp(
    depot: Dict[str, Any],
    stops: List[Dict[str, Any]],
    vehicles: List[Dict[str, int]],
    distance_matrix: Optional[List[List[float]]] = None,
    time_matrix: Optional[List[List[int]]] = None,
    time_limit_seconds: float = 30.0
) -> Dict[str, Any]:
    """
    Call VROOM VRP solver via HTTP API.
    
    Args:
        depot: Depot location with 'name', 'coordinates' [lat, lon], 'demand' (0)
        stops: List of stop locations, each with 'name', 'coordinates' [lat, lon], 'demand'
        vehicles: List of vehicles, each with 'capacity'
        distance_matrix: Optional pre-calculated NxN distance matrix (ignored, VROOM uses OSRM)
        time_matrix: Optional pre-calculated NxN time matrix (ignored, VROOM uses OSRM)
        time_limit_seconds: Maximum solving time (ignored for now)
    
    Returns:
        Response dictionary with routes and metrics
    """
    vroom_base_url = os.getenv("VROOM_BASE_URL", "http://localhost:3000/")
    url = vroom_base_url.rstrip("/") + "/"
    
    # Convert to VROOM format
    # VROOM uses [lon, lat] for coordinates
    depot_coords = depot["coordinates"]
    depot_lon, depot_lat = depot_coords[1], depot_coords[0]  # Convert [lat, lon] to [lon, lat]
    
    vroom_vehicles = [
        {
            "id": idx + 1,
            "start": [depot_lon, depot_lat],
            "end": [depot_lon, depot_lat],
            "capacity": [v["capacity"]]
        }
        for idx, v in enumerate(vehicles)
    ]
    
    vroom_jobs = [
        {
            "id": idx + 1,
            "location": [stop["coordinates"][1], stop["coordinates"][0]],  # [lon, lat]
            "service": 0,  # Service time in seconds (can be added later)
            "amount": [stop.get("demand", 0)]
        }
        for idx, stop in enumerate(stops)
    ]
    
    vroom_request = {
        "vehicles": vroom_vehicles,
        "jobs": vroom_jobs
    }
    
    logger.info(f"Calling VROOM with {len(stops)} stops and {len(vehicles)} vehicles")
    logger.debug(f"VROOM URL: {url}")
    logger.debug(f"VROOM request: {json.dumps(vroom_request, indent=2)}")
    
    try:
        # Call VROOM HTTP API
        req = urllib.request.Request(
            url,
            data=json.dumps(vroom_request).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=60) as response:
            vroom_response = json.loads(response.read().decode('utf-8'))
            
            if vroom_response.get("code") != 0:
                error_msg = f"VROOM solver error: code {vroom_response.get('code')}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # Convert VROOM response to our format
            return _convert_vroom_response(vroom_response, depot, stops, vehicles)
            
    except urllib.error.URLError as e:
        logger.warning(f"VROOM API call failed: {e}, using mock response")
        return _get_mock_vrp_response(stops, vehicles)
    except Exception as e:
        logger.warning(f"VROOM API error: {e}, using mock response")
        return _get_mock_vrp_response(stops, vehicles)


def _convert_vroom_response(
    vroom_response: Dict[str, Any],
    depot: Dict[str, Any],
    stops: List[Dict[str, Any]],
    vehicles: List[Dict[str, int]]
) -> Dict[str, Any]:
    """
    Convert VROOM response format to our standard format.
    
    VROOM format:
    {
        "code": 0,
        "summary": {"cost": ..., "service": ..., "duration": ..., "distance": ...},
        "routes": [{
            "vehicle": 1,
            "steps": [{"type": 0/1/2, "location": [lon, lat], "arrival": ..., "distance": ..., "id": ...}],
            "cost": ...,
            "duration": ...,
            "distance": ...
        }],
        "unassigned": [{"id": ...}]
    }
    
    Our format:
    {
        "routes": [{
            "vehicle_id": "V1",
            "ordered_stop_ids": ["S1", "S2", ...],
            "total_duration": ...,
            "total_distance": ...
        }],
        "metrics": {...}
    }
    """
    # Create mapping from VROOM job IDs to stop names
    job_to_stop = {idx + 1: stop["name"] for idx, stop in enumerate(stops)}
    vehicle_to_id = {idx + 1: f"V{idx + 1}" for idx in range(len(vehicles))}
    
    routes = []
    for vroom_route in vroom_response.get("routes", []):
        vehicle_id = vehicle_to_id.get(vroom_route.get("vehicle", 0), f"V{vroom_route.get('vehicle', 0)}")
        
        # Extract stop sequence from steps
        ordered_stop_ids = []
        for step in vroom_route.get("steps", []):
            step_type = step.get("type", -1)
            if step_type == 1:  # Job/stop (type 0=start, 1=job, 2=end)
                job_id = step.get("id")
                if job_id and job_id in job_to_stop:
                    ordered_stop_ids.append(job_to_stop[job_id])
        
        routes.append({
            "vehicle_id": vehicle_id,
            "ordered_stop_ids": ordered_stop_ids,
            "total_duration": vroom_route.get("duration", 0),
            "total_distance": vroom_route.get("distance", 0.0)
        })
    
    summary = vroom_response.get("summary", {})
    unassigned = vroom_response.get("unassigned", [])
    unassigned_stops = [job_to_stop.get(u.get("id", 0), f"unknown_{u.get('id')}") 
                       for u in unassigned if u.get("id") in job_to_stop]
    
    return {
        "routes": routes,
        "metrics": {
            "total_duration": summary.get("duration", 0),
            "total_distance": summary.get("distance", 0.0),
            "num_vehicles_used": len(routes),
            "unassigned_stops": unassigned_stops
        }
    }


def _get_mock_vrp_response(stops: List[Dict[str, Any]], vehicles: List[Dict[str, int]]) -> Dict[str, Any]:
    """Generate a mock VRP response for fallback."""
    # Simple round-robin assignment
    routes = []
    for v_idx, vehicle in enumerate(vehicles):
        vehicle_id = f"V{v_idx + 1}"
        # Assign stops in round-robin fashion
        assigned_stops = [stop["name"] for idx, stop in enumerate(stops) if idx % len(vehicles) == v_idx]
        if assigned_stops:
            routes.append({
                "vehicle_id": vehicle_id,
                "ordered_stop_ids": assigned_stops,
                "total_duration": len(assigned_stops) * 600,  # Mock: 10 min per stop
                "total_distance": len(assigned_stops) * 5000.0  # Mock: 5km per stop
            })
    
    return {
        "routes": routes,
        "metrics": {
            "total_duration": sum(r["total_duration"] for r in routes),
            "total_distance": sum(r["total_distance"] for r in routes),
            "num_vehicles_used": len(routes),
            "unassigned_stops": []
        }
    }


def parse_vrp_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse VRP response into standardized format.
    
    This function is kept for compatibility with existing code.
    The response from call_vrp() is already in the correct format.
    """
    return response

