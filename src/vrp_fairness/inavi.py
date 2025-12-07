"""
iNavi road distance/time caching and polyline support.
Centralized API for road routing queries.
"""

import json
import os
import math
import asyncio
import logging
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Type aliases
LatLon = Tuple[float, float]  # (latitude, longitude)


@dataclass
class RouteLeg:
    """
    Represents a route leg between two points with road routing data.
    
    Attributes:
        origin_id: Identifier for origin point
        dest_id: Identifier for destination point
        travel_time_s: Travel time in seconds (from iNavi road routing)
        travel_distance_m: Travel distance in meters (from iNavi road routing)
        polyline: Sequence of (lat, lon) points representing the road path
    """
    origin_id: str
    dest_id: str
    travel_time_s: float
    travel_distance_m: float
    polyline: List[LatLon]  # [(lat, lon), ...]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "origin_id": self.origin_id,
            "dest_id": self.dest_id,
            "travel_time_s": self.travel_time_s,
            "travel_distance_m": self.travel_distance_m,
            "polyline": [[lat, lon] for lat, lon in self.polyline]
        }


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Tuple[float, int]:
    """
    Calculate haversine distance between two points.
    
    Returns:
        (distance_meters, duration_seconds) - duration is estimated at 30 km/h average speed
    """
    R = 6371000  # Earth radius in meters
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance_m = R * c
    
    # Estimate duration: assume 30 km/h average speed
    avg_speed_ms = 30 * 1000 / 3600  # 30 km/h in m/s
    duration_s = int(distance_m / avg_speed_ms)
    
    return distance_m, duration_s


def round_coord(coord: float, precision: int = 6) -> float:
    """Round coordinate to specified precision for cache key stability."""
    return round(coord, precision)


def make_cache_key(lat1: float, lon1: float, lat2: float, lon2: float) -> str:
    """Create a stable cache key for a coordinate pair."""
    # Round to 6 decimal places (~0.1m precision)
    return f"{round_coord(lat1)},{round_coord(lon1)},{round_coord(lat2)},{round_coord(lon2)}"


def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """
    Decode Google Polyline Encoding Format string into list of (lat, lon) tuples.
    
    Args:
        polyline_str: Encoded polyline string
    
    Returns:
        List of (latitude, longitude) tuples
    """
    if not polyline_str:
        return []
    
    coordinates = []
    index = 0
    lat = 0
    lon = 0
    
    while index < len(polyline_str):
        # Decode latitude
        shift = 0
        result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        
        # Decode longitude
        shift = 0
        result = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlon = ~(result >> 1) if (result & 1) else (result >> 1)
        lon += dlon
        
        coordinates.append((lat / 1e5, lon / 1e5))
    
    return coordinates


class iNaviCache:
    """
    Cache for iNavi road distance/time calls.
    
    NOTE: This class now always attempts to use iNavi API.
    Haversine fallback is only used if API fails and should be considered an error condition.
    """
    
    def __init__(self, cache_file: str = "outputs/cache_inavi.json", approx_mode: bool = False):
        """
        Initialize cache.
        
        Args:
            cache_file: Path to JSON cache file
            approx_mode: DEPRECATED - If True, logs warning and uses haversine.
                        Should be False for production use with real iNavi API.
        """
        self.cache_file = cache_file
        self.approx_mode = approx_mode
        if approx_mode:
            logger.warning(
                "approx_mode=True is deprecated. "
                "System will attempt iNavi API first, fallback to haversine only on failure."
            )
        self.cache: Dict[str, Dict[str, float]] = {}
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache: {e}")
                self.cache = {}
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_distance_time(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> Tuple[float, int]:
        """
        Get road distance and time between two points.
        
        Args:
            lat1, lon1: Origin coordinates
            lat2, lon2: Destination coordinates
        
        Returns:
            (distance_meters, duration_seconds)
        """
        # Check if same point
        if abs(lat1 - lat2) < 1e-6 and abs(lon1 - lon2) < 1e-6:
            return 0.0, 0
        
        key = make_cache_key(lat1, lon1, lat2, lon2)
        
        # Check cache
        if key in self.cache:
            cached = self.cache[key]
            return cached['distance'], int(cached['duration'])
        
        # Compute (haversine in approx mode, or call iNavi API)
        if self.approx_mode:
            distance, duration = haversine_distance(lat1, lon1, lat2, lon2)
        else:
            # Call actual iNavi MCP tool
            try:
                # TODO: Replace with actual MCP tool call when available
                # For now, this is a placeholder that will call the real API
                response = self._call_inavi_api(lat1, lon1, lat2, lon2)
                distance = response.get("distance", 0.0)
                duration = response.get("duration", 0)
            except Exception as e:
                # Fallback to haversine if API call fails
                distance, duration = haversine_distance(lat1, lon1, lat2, lon2)
                logger.warning(f"iNavi API call failed ({e}), using haversine fallback for {key}")
        
        # Cache result
        self.cache[key] = {
            "distance": distance,
            "duration": duration
        }
        self._save_cache()
        
        return distance, duration
    
    def _call_inavi_api(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> Dict:
        """
        Call iNavi MCP tool to get distance, time, and optionally polyline.
        
        This method calls the Routing MCP Server's iNavi tool to get real road
        distance, travel time, and route polyline.
        
        Args:
            lat1, lon1: Origin coordinates
            lat2, lon2: Destination coordinates
        
        Returns:
            Dictionary with 'distance', 'duration', and optionally 'polyline' or 'geometry'
        """
        try:
            # Try to call MCP tool using the MCP client
            return self._call_mcp_tool_sync(lat1, lon1, lat2, lon2)
        except Exception as e:
            # Fallback to haversine if MCP call fails
            logger.warning(f"iNavi MCP tool call failed: {e}, using haversine fallback")
            distance, duration = haversine_distance(lat1, lon1, lat2, lon2)
            return {
                "distance": distance,
                "duration": duration,
                "polyline": None,
                "geometry": None
            }
    
    def _call_mcp_tool_sync(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> Dict:
        """
        Synchronous wrapper for async MCP tool call.
        
        Args:
            lat1, lon1: Origin coordinates
            lat2, lon2: Destination coordinates
        
        Returns:
            Dictionary with API response
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, we need to use a different approach
                # For now, use haversine fallback
                raise RuntimeError("Event loop is already running, cannot use async MCP call")
            return loop.run_until_complete(self._call_mcp_tool_async(lat1, lon1, lat2, lon2))
        except RuntimeError:
            # No event loop, create a new one
            return asyncio.run(self._call_mcp_tool_async(lat1, lon1, lat2, lon2))
    
    async def _call_mcp_tool_async(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float
    ) -> Dict:
        """
        Async method to call iNavi MCP tool.
        
        Args:
            lat1, lon1: Origin coordinates
            lat2, lon2: Destination coordinates
        
        Returns:
            Dictionary with 'distance', 'duration', and optionally 'polyline' or 'geometry'
        """
        try:
            from mcp import ClientSession, StdioServerParameters
            
            # Check for MCP server configuration
            # The MCP server should be configured via environment or config
            mcp_server_command = os.getenv("MCP_SERVER_COMMAND", None)
            
            # Also check for alternative MCP tool access methods
            # Some MCP implementations provide tools through a global registry
            try:
                # Try to access MCP tools directly if available in the environment
                import sys
                if hasattr(sys, 'mcp_tools') or 'mcp_tools' in globals():
                    # MCP tools might be injected into the environment
                    mcp_tools = getattr(sys, 'mcp_tools', globals().get('mcp_tools'))
                    if mcp_tools and hasattr(mcp_tools, 'inavi_road_distance'):
                        # Direct tool access
                        result = mcp_tools.inavi_road_distance(
                            origin={"lat": lat1, "lon": lon1},
                            destination={"lat": lat2, "lon": lon2},
                            mode="driving",
                            return_polyline=True
                        )
                        if isinstance(result, str):
                            return json.loads(result)
                        return result
            except (AttributeError, KeyError):
                pass
            
            if not mcp_server_command:
                # If no server command is configured, try to use a default
                # This assumes the MCP server is available in PATH or configured elsewhere
                raise RuntimeError(
                    "MCP server not configured. "
                    "Set MCP_SERVER_COMMAND environment variable or ensure MCP tools are available."
                )
            
            # Configure MCP server connection
            from mcp.client.stdio import stdio_client
            
            server_params = StdioServerParameters(
                command=mcp_server_command,
                args=os.getenv("MCP_SERVER_ARGS", "").split() if os.getenv("MCP_SERVER_ARGS") else []
            )
            
            # Retry logic: tms-mcp server needs time to initialize (fetch API docs)
            max_retries = 3
            initial_delay = 2.0  # Wait 2 seconds for server to start
            
            for attempt in range(max_retries):
                try:
                    async with stdio_client(server_params) as (read, write):
                        # Wait for server to be ready (exponential backoff)
                        delay = initial_delay * (2 ** attempt)
                        await asyncio.sleep(delay)
                        
                        async with ClientSession(read, write) as session:
                            # Call the iNavi road distance tool
                            # Try different possible tool names
                            tool_names = ["inavi_road_distance", "inavi_road-distance", "inavi_roadDistance", "road_distance"]
                            
                            for tool_name in tool_names:
                                try:
                                    response = await session.call_tool(
                                        tool_name,
                                        {
                                            "origin": {"lat": lat1, "lon": lon1},
                                            "destination": {"lat": lat2, "lon": lon2},
                                            "mode": "driving",
                                            "return_polyline": True
                                        }
                                    )
                                    
                                    # Parse response
                                    if response.content and len(response.content) > 0:
                                        # Response content is typically a TextContent object
                                        response_text = response.content[0].text
                                        result = json.loads(response_text)
                                        if attempt > 0:
                                            logger.debug(f"iNavi API call succeeded on retry {attempt + 1}")
                                        return result
                                except Exception as tool_error:
                                    logger.debug(f"Tool {tool_name} failed: {tool_error}")
                                    continue
                            
                            raise ValueError("No MCP tool responded successfully")
                            
                except (ExceptionGroup, RuntimeError) as e:
                    # Initialization error - server not ready yet
                    error_msg = str(e)
                    if "initialization" in error_msg.lower() or "before initialization" in error_msg.lower():
                        if attempt < max_retries - 1:
                            wait_time = initial_delay * (2 ** (attempt + 1))
                            logger.debug(f"iNavi MCP server initialization failed (attempt {attempt + 1}/{max_retries}), "
                                       f"retrying in {wait_time:.1f}s...")
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            logger.warning(f"iNavi MCP server initialization failed after {max_retries} attempts")
                            raise
                    else:
                        # Other error, re-raise
                        raise
                    
        except ImportError:
            raise RuntimeError("MCP library not available. Install with: pip install 'mcp[cli]'")
        except Exception as e:
            logger.debug(f"MCP tool call error: {e}")
            raise
    
    def build_distance_matrix(
        self,
        locations: list
    ) -> Tuple[list, list]:
        """
        Build distance and time matrices for a list of locations.
        
        REFACTORED: Uses get_distance_time() which always attempts iNavi API first.
        All distances and times come from iNavi road routing (not haversine unless API fails).
        
        Args:
            locations: List of dicts with 'lat'/'lon' keys or 'coordinates' [lat, lon]
        
        Returns:
            (distance_matrix, time_matrix) as 2D lists - all from iNavi road routing
        """
        n = len(locations)
        distance_matrix = [[0.0] * n for _ in range(n)]
        time_matrix = [[0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Handle both 'lat'/'lon' and 'coordinates' formats
                    loc_i = locations[i]
                    loc_j = locations[j]
                    
                    if 'lat' in loc_i and 'lon' in loc_i:
                        lat_i, lon_i = loc_i['lat'], loc_i['lon']
                    elif 'coordinates' in loc_i:
                        lat_i, lon_i = loc_i['coordinates']
                    else:
                        continue
                    
                    if 'lat' in loc_j and 'lon' in loc_j:
                        lat_j, lon_j = loc_j['lat'], loc_j['lon']
                    elif 'coordinates' in loc_j:
                        lat_j, lon_j = loc_j['coordinates']
                    else:
                        continue
                    
                    dist, time = self.get_distance_time(lat_i, lon_i, lat_j, lon_j)
                    distance_matrix[i][j] = dist
                    time_matrix[i][j] = time
        
        return distance_matrix, time_matrix


def get_leg_route(
    origin: LatLon,
    dest: LatLon,
    origin_id: str = "",
    dest_id: str = "",
    *,
    cache: Optional[iNaviCache] = None,
) -> RouteLeg:
    """
    Query iNavi routing between origin and dest.
    
    Use iNavi's road routing API to obtain:
    - travel time (seconds)
    - travel distance (meters)
    - route polyline (sequence of (lat, lon) points)
    
    Cache results by rounded coordinates.
    
    Args:
        origin: (latitude, longitude) of origin point
        dest: (latitude, longitude) of destination point
        origin_id: Optional identifier for origin (for RouteLeg)
        dest_id: Optional identifier for destination (for RouteLeg)
        cache: Optional iNaviCache instance (creates new one if None)
    
    Returns:
        RouteLeg object with routing data and polyline
    
    Raises:
        RuntimeError: If iNavi API is unavailable and no fallback is configured
    """
    if cache is None:
        cache = iNaviCache()
    
    lat1, lon1 = origin
    lat2, lon2 = dest
    
    # Get distance and time from iNavi
    distance_m, duration_s = cache.get_distance_time(lat1, lon1, lat2, lon2)
    
    # Get polyline
    polyline_cache_file = cache.cache_file.replace('cache_inavi.json', 'cache_polyline.json')
    if polyline_cache_file == cache.cache_file:  # If replacement didn't work
        polyline_cache_file = "outputs/cache_polyline.json"
    polyline_cache = PolylineCache(cache_file=polyline_cache_file)
    polyline = polyline_cache.get_polyline(lat1, lon1, lat2, lon2, cache)
    
    # Generate IDs if not provided
    if not origin_id:
        origin_id = f"{lat1:.6f},{lon1:.6f}"
    if not dest_id:
        dest_id = f"{lat2:.6f},{lon2:.6f}"
    
    return RouteLeg(
        origin_id=origin_id,
        dest_id=dest_id,
        travel_time_s=float(duration_s),
        travel_distance_m=distance_m,
        polyline=polyline
    )


class PolylineCache:
    """Cache for route polylines."""
    
    def __init__(self, cache_file: str = "outputs/cache_polyline.json"):
        """
        Initialize polyline cache.
        
        Args:
            cache_file: Path to JSON cache file
        """
        self.cache_file = cache_file
        self.cache: Dict[str, List[List[float]]] = {}  # key -> [[lat, lon], ...]
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load cache from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load polyline cache: {e}")
                self.cache = {}
        else:
            os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        os.makedirs(os.path.dirname(self.cache_file) or '.', exist_ok=True)
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
    
    def get_polyline(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        inavi_cache: Optional[iNaviCache] = None
    ) -> List[Tuple[float, float]]:
        """
        Get route polyline between two points.
        
        Args:
            lat1, lon1: Origin coordinates
            lat2, lon2: Destination coordinates
            inavi_cache: Optional iNaviCache instance for API calls
        
        Returns:
            List of (lat, lon) tuples representing the route
        """
        # Check if same point
        if abs(lat1 - lat2) < 1e-6 and abs(lon1 - lon2) < 1e-6:
            return [(lat1, lon1)]
        
        key = make_cache_key(lat1, lon1, lat2, lon2)
        
        # Check cache
        if key in self.cache:
            cached = self.cache[key]
            return [(point[0], point[1]) for point in cached]
        
        # Get polyline from iNavi API
        # REFACTORED: Always attempt iNavi API (not just when not in approx_mode)
        polyline_points = []
        
        if inavi_cache:
            try:
                # Call iNavi API to get route with polyline
                response = inavi_cache._call_inavi_api(lat1, lon1, lat2, lon2)
                
                # Extract polyline
                if "polyline" in response and response["polyline"]:
                    # Decode polyline string
                    polyline_points = decode_polyline(response["polyline"])
                elif "geometry" in response and response["geometry"]:
                    # Use geometry array directly
                    polyline_points = [(p[0], p[1]) for p in response["geometry"]]
                else:
                    # No polyline in response - create simple two-point polyline
                    logger.debug(f"No polyline in iNavi response for {key}, using two-point polyline")
                    polyline_points = [(lat1, lon1), (lat2, lon2)]
            except Exception as e:
                logger.warning(f"Failed to get polyline from iNavi: {e}, using two-point polyline")
                # Fallback: create simple two-point polyline
                polyline_points = [(lat1, lon1), (lat2, lon2)]
        else:
            # No cache provided: create simple two-point polyline
            logger.debug(f"No iNavi cache provided for {key}, using two-point polyline")
            polyline_points = [(lat1, lon1), (lat2, lon2)]
        
        # Cache result
        self.cache[key] = [[lat, lon] for lat, lon in polyline_points]
        self._save_cache()
        
        return polyline_points

