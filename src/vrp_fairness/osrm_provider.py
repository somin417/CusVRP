"""
OSRM-based time/distance providers and polyline fetcher (no iNavi).
"""

import os
import logging
from typing import Dict, List, Any, Callable, Optional, Tuple

import requests

from .inavi import iNaviCache, PolylineCache, make_cache_key
from .geometry import normalize_geometry_to_latlon

logger = logging.getLogger(__name__)


OSRM_BASE_URL = os.environ.get("OSRM_BASE_URL", "http://localhost:5001")


def create_osrm_providers(
    depots: List[Dict[str, Any]],
    stops_by_id: Dict[str, Dict[str, Any]],
    cache: Optional[iNaviCache] = None,
    poly_cache: Optional[PolylineCache] = None
) -> Tuple[Callable[[str, str], float], Callable[[str, str], float]]:
    """
    Create time and distance provider functions using OSRM.
    
    Args:
        depots: List of depot dicts with 'id', 'lat', 'lon'
        stops_by_id: Dict mapping stop_id -> stop data with 'lat', 'lon'
        cache: Optional cache for OSRM lookups
    
    Returns:
        (time_provider, distance_provider) where:
        - time_provider(from_id, to_id) -> travel_time_seconds
        - distance_provider(from_id, to_id) -> distance_meters
    """
    if cache is None:
        cache = iNaviCache(approx_mode=False)  # Use OSRM, not haversine
    if poly_cache is None:
        poly_cache = PolylineCache(cache_file="outputs/cache_osrm_polyline.json")
    
    # Build location lookup
    locations = {}
    for depot in depots:
        locations[depot["id"]] = (depot["lat"], depot["lon"])
    for stop_id, stop_data in stops_by_id.items():
        locations[stop_id] = (stop_data["lat"], stop_data["lon"])
    
    def time_provider(from_id: str, to_id: str) -> float:
        """Get OSRM travel time between two locations."""
        if from_id not in locations or to_id not in locations:
            logger.warning(f"Missing location: {from_id} or {to_id}")
            return 0.0
        
        lat1, lon1 = locations[from_id]
        lat2, lon2 = locations[to_id]
        
        leg = get_osrm_leg((lat1, lon1), (lat2, lon2), cache=cache, poly_cache=poly_cache)
        return leg["duration_s"]
    
    def distance_provider(from_id: str, to_id: str) -> float:
        """Get OSRM distance between two locations."""
        if from_id not in locations or to_id not in locations:
            logger.warning(f"Missing location: {from_id} or {to_id}")
            return 0.0
        
        lat1, lon1 = locations[from_id]
        lat2, lon2 = locations[to_id]
        
        leg = get_osrm_leg((lat1, lon1), (lat2, lon2), cache=cache, poly_cache=poly_cache)
        return leg["distance_m"]
    
    return time_provider, distance_provider


def get_osrm_leg(
    origin: Tuple[float, float],
    dest: Tuple[float, float],
    *,
    cache: Optional[iNaviCache] = None,
    poly_cache: Optional[PolylineCache] = None
) -> Dict[str, Any]:
    """
    Fetch OSRM route leg (distance, duration, polyline) between two coordinates.
    Uses PolylineCache to avoid duplicate calls.
    """
    lat1, lon1 = origin
    lat2, lon2 = dest
    
    if cache is None:
        cache = iNaviCache(approx_mode=False)
    
    if poly_cache is None:
        poly_cache = PolylineCache(cache_file="outputs/cache_osrm_polyline.json")
    key = make_cache_key(lat1, lon1, lat2, lon2)
    
    # Try cache first
    cached_poly = poly_cache.cache.get(key)
    if cached_poly:
        normalized = normalize_geometry_to_latlon(cached_poly)
    else:
        normalized = None
    
    # Call OSRM for fresh values
    url = f"{OSRM_BASE_URL}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "overview": "full",
        "geometries": "geojson"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        route = data["routes"][0]
        distance_m = float(route.get("distance", 0.0))
        duration_s = float(route.get("duration", 0.0))
        geometry_coords = route.get("geometry", {}).get("coordinates", [])
        
        # OSRM returns [lon, lat]; normalize to [(lat, lon)]
        polyline_coords = [(lat, lon) for lon, lat in geometry_coords]
        if polyline_coords:
            poly_cache.cache[key] = polyline_coords
            poly_cache._save_cache()
            normalized = polyline_coords
    except Exception as e:
        logger.warning(f"OSRM route call failed for {key}: {e}")
        # Fallback to haversine distance/time via cache.get_distance_time
        distance_m, duration_s = cache.get_distance_time(lat1, lon1, lat2, lon2)
    
    return {
        "origin": origin,
        "dest": dest,
        "distance_m": distance_m if 'distance_m' in locals() else 0.0,
        "duration_s": duration_s if 'duration_s' in locals() else 0.0,
        "polyline": normalized or []
    }

