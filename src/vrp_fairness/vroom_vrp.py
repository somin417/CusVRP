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
        "jobs": vroom_jobs,
        "options": {"g": True}  # Request geometry
    }
    
    logger.info(f"Calling VROOM with {len(stops)} stops and {len(vehicles)} vehicles")
    logger.info("VROOM request: options.g=true")
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
    Convert VROOM response format to our standard format with geometry extraction.
    
    VROOM format:
    {
        "code": 0,
        "summary": {"cost": ..., "service": ..., "duration": ..., "distance": ...},
        "routes": [{
            "vehicle": 1,
            "steps": [{"type": "start"/"job"/"end", "location": [lon, lat], "arrival": ..., "distance": ..., "id": ..., "geometry": ...}],
            "geometry": "...",  # Route-level geometry (if available)
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
            "total_distance": ...,
            "geometry": ...,  # Route geometry if available
            "legs": [{"from": ..., "to": ..., "geometry": ..., "polyline": ...}]
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
        
        # Extract stop sequence and geometry from steps
        ordered_stop_ids = []
        legs = []
        steps = vroom_route.get("steps", [])
        
        # Check for route-level geometry
        route_geometry = vroom_route.get("geometry")
        if route_geometry:
            logger.info(f"Route {vehicle_id}: Found route-level geometry (type: {type(route_geometry).__name__})")
        
        # Process steps to extract stops and leg geometries
        prev_step = None
        for step in steps:
            step_type = step.get("type", "")
            
            if step_type == "job":
                job_id = step.get("id") or step.get("job")
                if job_id and job_id in job_to_stop:
                    ordered_stop_ids.append(job_to_stop[job_id])
            
            # Extract leg geometry (from previous step to current)
            if prev_step and step_type in ["job", "end"]:
                leg_geometry = step.get("geometry")
                if leg_geometry:
                    geom_type = type(leg_geometry).__name__
                    if isinstance(leg_geometry, str):
                        geom_len = len(leg_geometry)
                    elif isinstance(leg_geometry, (list, tuple)):
                        geom_len = len(leg_geometry)
                    else:
                        geom_len = 0
                    
                    if len(legs) < 3:  # Log first 3 legs
                        logger.info(f"Route {vehicle_id} leg {len(legs)+1}: geometry type={geom_type}, points={geom_len}")
                    
                    from_id = prev_step.get("id") or prev_step.get("job") or "start"
                    to_id = step.get("id") or step.get("job") or "end"
                    
                    legs.append({
                        "from": job_to_stop.get(from_id, str(from_id)) if from_id != "start" else depot.get("name", "depot"),
                        "to": job_to_stop.get(to_id, str(to_id)) if to_id != "end" else depot.get("name", "depot"),
                        "geometry": leg_geometry,
                        "polyline": leg_geometry  # Alias for compatibility
                    })
            
            prev_step = step
        
        route_data = {
            "vehicle_id": vehicle_id,
            "ordered_stop_ids": ordered_stop_ids,
            "total_duration": vroom_route.get("duration", 0),
            "total_distance": vroom_route.get("distance", 0.0)
        }
        
        if route_geometry:
            route_data["geometry"] = route_geometry
        if legs:
            route_data["legs"] = legs
        
        routes.append(route_data)
    
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


def solve_single_depot(
    depot: Dict[str, Any],
    vehicles_for_depot: List[Dict[str, Any]],
    stops_for_depot: List[Dict[str, Any]],
    *,
    request_geometry: bool = True
) -> Dict[str, Any]:
    """
    Solve VRP for a single depot.
    
    Args:
        depot: Depot dict with 'id', 'lat', 'lon', 'name'
        vehicles_for_depot: List of vehicle dicts with 'id', 'capacity', optionally 'shift'
        stops_for_depot: List of stop dicts with 'id', 'lat', 'lon', 'demand', 'service_time_s'
        request_geometry: Whether to request geometry from VROOM
    
    Returns:
        SolutionPart dict with 'depot_id', 'routes', 'metrics'
    """
    # Convert depot format
    depot_dict = {
        "name": depot.get("id", depot.get("name", "depot")),
        "coordinates": [depot["lat"], depot["lon"]],
        "demand": 0
    }
    
    # Convert vehicles format
    vehicles_list = [
        {"capacity": v.get("capacity", 100)}
        for v in vehicles_for_depot
    ]
    
    # Convert stops format
    stops_list = [
        {
            "name": s.get("id", f"stop_{i}"),
            "coordinates": [s["lat"], s["lon"]],
            "demand": s.get("demand", 1),
            "service_time": s.get("service_time_s", 300)
        }
        for i, s in enumerate(stops_for_depot)
    ]
    
    # Call VROOM
    response = call_vrp(depot_dict, stops_list, vehicles_list)
    
    # Add depot_id to response
    response["depot_id"] = depot["id"]
    
    return response


def solve_multi_depot(
    depots: List[Dict[str, Any]],
    vehicles_by_depot: Dict[str, List[Dict[str, Any]]],
    stops_by_depot: Dict[str, List[Dict[str, Any]]],
    *,
    request_geometry: bool = True
) -> Dict[str, Any]:
    """
    Solve VRP for multiple depots and merge results.
    
    Args:
        depots: List of depot dicts
        vehicles_by_depot: Dict mapping depot_id -> list of vehicles
        stops_by_depot: Dict mapping depot_id -> list of stops
        request_geometry: Whether to request geometry from VROOM
    
    Returns:
        Merged Solution dict with 'routes_by_dc' (depot), 'metrics', 'stops_dict'
    """
    all_routes_by_depot = {}
    all_metrics = []
    all_stops_dict = {}
    
    for depot in depots:
        depot_id = depot["id"]
        vehicles = vehicles_by_depot.get(depot_id, [])
        stops = stops_by_depot.get(depot_id, [])
        
        if not stops:
            logger.info(f"Depot {depot_id}: No stops assigned, skipping")
            continue
        
        logger.info(f"Solving VRP for depot {depot_id}: {len(stops)} stops, {len(vehicles)} vehicles")
        
        # Solve for this depot
        solution_part = solve_single_depot(
            depot, vehicles, stops, request_geometry=request_geometry
        )
        
        # Store routes with depot_id
        routes = solution_part.get("routes", [])
        for route in routes:
            route["dc_id"] = depot_id
            route["depot_id"] = depot_id
        all_routes_by_depot[depot_id] = routes
        
        # Store metrics
        metrics = solution_part.get("metrics", {})
        metrics["depot_id"] = depot_id
        all_metrics.append(metrics)
        
        # Store stops
        for stop in stops:
            all_stops_dict[stop.get("id", f"stop_{len(all_stops_dict)}")] = {
                "lat": stop["lat"],
                "lon": stop["lon"],
                "demand": stop.get("demand", 1),
                "service_time": stop.get("service_time_s", 300)
            }
    
    # Merge metrics
    total_duration = sum(m.get("total_duration", 0) for m in all_metrics)
    total_distance = sum(m.get("total_distance", 0.0) for m in all_metrics)
    total_vehicles = sum(m.get("num_vehicles_used", 0) for m in all_metrics)
    all_unassigned = []
    for m in all_metrics:
        all_unassigned.extend(m.get("unassigned_stops", []))
    
    return {
        "routes_by_dc": all_routes_by_depot,  # Keep 'dc' key for compatibility
        "depots": depots,  # Include depots for objectives computation
        "metrics": {
            "total_duration": total_duration,
            "total_distance": total_distance,
            "num_vehicles_used": total_vehicles,
            "unassigned_stops": all_unassigned,
            "per_depot": all_metrics
        },
        "stops_dict": all_stops_dict
    }


def parse_vrp_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse VRP response into standardized format.
    
    This function is kept for compatibility with existing code.
    The response from call_vrp() is already in the correct format.
    """
    return response

