"""
OSRM-based time and distance providers (no iNavi).
"""

import logging
from typing import Dict, List, Any, Callable, Optional, Tuple
from .inavi import get_leg_route, iNaviCache

logger = logging.getLogger(__name__)


def create_osrm_providers(
    depots: List[Dict[str, Any]],
    stops_by_id: Dict[str, Dict[str, Any]],
    cache: Optional[iNaviCache] = None
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
        
        leg = get_leg_route(
            origin=(lat1, lon1),
            dest=(lat2, lon2),
            cache=cache
        )
        return leg.travel_time_s
    
    def distance_provider(from_id: str, to_id: str) -> float:
        """Get OSRM distance between two locations."""
        if from_id not in locations or to_id not in locations:
            logger.warning(f"Missing location: {from_id} or {to_id}")
            return 0.0
        
        lat1, lon1 = locations[from_id]
        lat2, lon2 = locations[to_id]
        
        leg = get_leg_route(
            origin=(lat1, lon1),
            dest=(lat2, lon2),
            cache=cache
        )
        return leg.travel_distance_m
    
    return time_provider, distance_provider

