"""
Geometry parsing and normalization for OSRM/VROOM routes.
"""

import logging
from typing import List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

try:
    import polyline
    POLYLINE_AVAILABLE = True
except ImportError:
    POLYLINE_AVAILABLE = False
    logger.warning("polyline package not available, encoded polylines cannot be decoded")


def normalize_geometry_to_latlon(geom: Any) -> Optional[List[Tuple[float, float]]]:
    """
    Normalize geometry to list of (lat, lon) tuples.
    
    Handles:
    - Encoded polyline strings (OSRM format)
    - GeoJSON LineString coordinates [[lon,lat], ...] or [[lat,lon], ...]
    - List of [lat, lon] or [lon, lat] pairs
    
    Args:
        geom: Geometry in various formats
    
    Returns:
        List of (lat, lon) tuples, or None if parsing fails
    """
    if geom is None:
        return None
    
    # Case 1: Encoded polyline string
    if isinstance(geom, str):
        if not POLYLINE_AVAILABLE:
            logger.warning("Encoded polyline detected but polyline package not available")
            return None
        
        try:
            decoded = polyline.decode(geom)
            logger.debug(f"Decoded polyline string: {len(decoded)} points")
            # polyline.decode returns [(lat, lon), ...]
            return decoded
        except Exception as e:
            logger.warning(f"Failed to decode polyline: {e}")
            return None
    
    # Case 2: List/array of coordinates
    if isinstance(geom, (list, tuple)):
        if len(geom) == 0:
            return None
        
        # Check first element to determine format
        first = geom[0]
        
        # GeoJSON format: [[lon, lat], ...] or [[lat, lon], ...]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            # Detect coordinate order by checking if lon is in valid range
            # Longitude for Korea: ~127-130, Latitude: ~33-38
            first_lon, first_lat = first[0], first[1]
            
            # If first value is > 100, likely [lon, lat] (GeoJSON)
            # If first value is < 50, likely [lat, lon]
            if first_lon > 100 or first_lon < -180:
                # Likely [lon, lat] format - swap
                logger.debug(f"Detected [lon, lat] format, converting to [lat, lon]")
                return [(float(coord[1]), float(coord[0])) for coord in geom if len(coord) >= 2]
            else:
                # Likely [lat, lon] format
                logger.debug(f"Detected [lat, lon] format")
                return [(float(coord[0]), float(coord[1])) for coord in geom if len(coord) >= 2]
        
        # Flat list: assume [lat, lon, lat, lon, ...] or [lon, lat, lon, lat, ...]
        if len(geom) >= 2 and isinstance(first, (int, float)):
            # Check first value to determine order
            if abs(first) > 100:
                # Likely [lon, lat, lon, lat, ...]
                logger.debug(f"Detected flat [lon, lat, ...] format")
                return [(float(geom[i+1]), float(geom[i])) for i in range(0, len(geom)-1, 2)]
            else:
                # Likely [lat, lon, lat, lon, ...]
                logger.debug(f"Detected flat [lat, lon, ...] format")
                return [(float(geom[i]), float(geom[i+1])) for i in range(0, len(geom)-1, 2)]
    
    logger.warning(f"Unknown geometry format: {type(geom)}")
    return None

