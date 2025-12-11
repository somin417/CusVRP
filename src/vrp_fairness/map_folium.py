"""
Folium-based map visualization for VRP routes.
"""

import logging
from typing import Dict, List, Any, Optional, Literal
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import folium
    from folium import plugins
except ImportError:
    folium = None
    logger.warning("folium not installed. Install with: pip install folium")

# Import for geometry calculation (OSRM)
try:
    from .osrm_provider import get_osrm_leg
    from .inavi import iNaviCache
    OSRM_AVAILABLE = True
except ImportError:
    OSRM_AVAILABLE = False
    logger.warning("OSRM geometry calculation not available")


def _extract_polyline_coords(polyline) -> List[tuple]:
    """Extract coordinates from polyline (list or string)."""
    leg_coords = []
    
    if isinstance(polyline, str):
        # Encoded polyline string
        leg_coords = decode_polyline_string(polyline)
    elif isinstance(polyline, list) and len(polyline) > 0:
        if isinstance(polyline[0], (list, tuple)):
            # List of [lat, lon] or [lon, lat]
            for point in polyline:
                if len(point) >= 2:
                    # Assume [lat, lon] format (check if reasonable lat)
                    if abs(point[0]) <= 90:  # Likely lat
                        leg_coords.append((float(point[0]), float(point[1])))
                    else:  # Likely [lon, lat]
                        leg_coords.append((float(point[1]), float(point[0])))
        else:
            # Single coordinate pair
            if len(polyline) >= 2:
                if abs(polyline[0]) <= 90:
                    leg_coords.append((float(polyline[0]), float(polyline[1])))
                else:
                    leg_coords.append((float(polyline[1]), float(polyline[0])))
    
    return leg_coords


def decode_polyline_string(polyline_str: str) -> List[tuple]:
    """
    Decode encoded polyline string to list of (lat, lon) tuples.
    
    Args:
        polyline_str: Encoded polyline string (e.g., from OSRM)
    
    Returns:
        List of (lat, lon) tuples
    """
    try:
        import polyline as polyline_lib
        return polyline_lib.decode(polyline_str)
    except ImportError:
        logger.warning("polyline package not installed. Cannot decode polyline strings.")
        return []
    except Exception as e:
        logger.warning(f"Failed to decode polyline: {e}")
        return []


def extract_route_coordinates(
    route: Dict[str, Any],
    stops_by_id: Dict[str, Dict[str, Any]],
    dc: Any
) -> List[tuple]:
    """
    Extract route coordinates from route data, prioritizing OSRM geometry.
    
    Priority:
    1. Route-level geometry (if available)
    2. Per-leg geometry (concatenated)
    3. Fallback to straight lines through stops
    
    Args:
        route: Route dict with 'ordered_stop_ids' and optionally 'geometry' or 'legs'
        stops_by_id: Dict mapping stop ID to stop data with 'lat', 'lon'
        dc: DC config with 'lat', 'lon', 'id'
    
    Returns:
        List of (lat, lon) tuples for the route
    """
    from .geometry import normalize_geometry_to_latlon
    
    coordinates = []
    stop_ids = route.get("ordered_stop_ids", [])
    
    # Priority 1: Route-level geometry
    # BUT: Only use if it matches the current stop order
    # (Improved routes may have old geometry from baseline)
    route_geometry = route.get("geometry")
    if route_geometry:
        # Check if this is likely an improved route with mismatched geometry
        # If route has dc_id and we're in improved layer, be more cautious
        # For now, always validate geometry matches stop order by checking first/last points
        normalized = normalize_geometry_to_latlon(route_geometry)
        if normalized and len(normalized) > 20:
            # Quick validation: check if geometry endpoints match route endpoints
            stop_ids = route.get("ordered_stop_ids", [])
            if stop_ids and stops_by_id:
                first_stop = stops_by_id.get(stop_ids[0])
                last_stop = stops_by_id.get(stop_ids[-1])
                if first_stop and last_stop:
                    # Check if geometry starts/ends near first/last stops
                    geom_start = normalized[0]
                    geom_end = normalized[-1]
                    start_dist = ((geom_start[0] - first_stop["lat"])**2 + (geom_start[1] - first_stop["lon"])**2)**0.5
                    end_dist = ((geom_end[0] - last_stop["lat"])**2 + (geom_end[1] - last_stop["lon"])**2)**0.5
                    
                    # If geometry doesn't match stop locations, skip it
                    if start_dist > 0.01 or end_dist > 0.01:  # ~1km threshold
                        logger.warning(f"Geometry endpoints don't match route stops, recalculating from stop order")
                        route_geometry = None  # Force fallback to stop order
                    else:
                        logger.info(f"DRAWING GEOMETRY route-level polyline points={len(normalized)}")
                        return normalized
            
            if normalized and len(normalized) > 20:
                logger.info(f"DRAWING GEOMETRY route-level polyline points={len(normalized)}")
                return normalized
        elif normalized:
            logger.warning(f"Route geometry has only {len(normalized)} points, trying legs")
    
    # Priority 2: Per-leg geometry (concatenate all legs)
    if "legs" in route and route["legs"]:
        all_leg_coords = []
        total_points = 0
        
        for leg in route["legs"]:
            leg_geom = leg.get("geometry") or leg.get("polyline")
            if leg_geom:
                normalized = normalize_geometry_to_latlon(leg_geom)
                if normalized:
                    total_points += len(normalized)
                    all_leg_coords.extend(normalized)
        
        if total_points > 20:
            logger.info(f"DRAWING GEOMETRY per-leg polyline points={total_points}")
            return all_leg_coords
        elif total_points > 0:
            logger.warning(f"Per-leg geometry has only {total_points} points, using fallback")
        
        # Build mapping of leg endpoints for easier lookup (fallback)
        leg_map = {}
        for leg in route["legs"]:
            from_id = leg.get("from")
            to_id = leg.get("to")
            if from_id and to_id:
                leg_map[(from_id, to_id)] = leg
        
        # Process route: DC -> first stop -> ... -> last stop -> DC
        if stop_ids:
            # Start from DC (only once)
            if not coordinates or coordinates[-1] != (dc.lat, dc.lon):
                coordinates.append((dc.lat, dc.lon))
            
            # Process each stop in sequence, inserting exact stop coordinates
            for stop_idx, stop_id in enumerate(stop_ids):
                if stop_id not in stops_by_id:
                    logger.warning(f"Stop {stop_id} not in stops_by_id, skipping")
                    continue
                
                stop = stops_by_id[stop_id]
                stop_coord = (stop["lat"], stop["lon"])
                
                # Get leg polyline (road path) to this stop
                if stop_idx == 0:
                    # First stop: DC -> stop
                    leg_key = (dc.id, stop_id)
                    prev_coord = (dc.lat, dc.lon)
                else:
                    # Subsequent stops: previous stop -> this stop
                    leg_key = (stop_ids[stop_idx - 1], stop_id)
                    prev_stop = stops_by_id[stop_ids[stop_idx - 1]]
                    prev_coord = (prev_stop["lat"], prev_stop["lon"])
                
                leg = leg_map.get(leg_key)
                if leg:
                    leg_geom = leg.get("geometry") or leg.get("polyline")
                    leg_coords = normalize_geometry_to_latlon(leg_geom) if leg_geom else []
                    if leg_coords and len(leg_coords) > 1:
                        # Add intermediate road path coordinates
                        # Include all points except exact duplicates of start/end
                        for coord in leg_coords:
                            # Skip if it's the exact previous coordinate
                            if coordinates and abs(coordinates[-1][0] - coord[0]) < 1e-8 and abs(coordinates[-1][1] - coord[1]) < 1e-8:
                                continue
                            # Skip if it's the exact stop coordinate (we'll add it explicitly)
                            if abs(coord[0] - stop_coord[0]) < 1e-8 and abs(coord[1] - stop_coord[1]) < 1e-8:
                                continue
                            coordinates.append(coord)
                    elif leg_coords and len(leg_coords) == 1:
                        # Only one point in leg - add it if not duplicate
                        coord = leg_coords[0]
                        if coordinates and (abs(coordinates[-1][0] - coord[0]) > 1e-8 or abs(coordinates[-1][1] - coord[1]) > 1e-8):
                            if abs(coord[0] - stop_coord[0]) > 1e-8 or abs(coord[1] - stop_coord[1]) > 1e-8:
                                coordinates.append(coord)
                
                # Always add interpolated points if distance is large (for visual continuity)
                prev_coord = coordinates[-1] if coordinates else prev_coord
                dist = ((stop_coord[0] - prev_coord[0])**2 + (stop_coord[1] - prev_coord[1])**2)**0.5
                # If distance is large (>0.0005 degrees ~50m), add intermediate points
                if dist > 0.0005:
                    # Add intermediate points for visual continuity (more points for longer distances)
                    num_interp = min(8, max(3, int(dist / 0.0002)))  # 3-8 points based on distance
                    for i in range(1, num_interp + 1):
                        ratio = i / (num_interp + 1)
                        interp_lat = prev_coord[0] + (stop_coord[0] - prev_coord[0]) * ratio
                        interp_lon = prev_coord[1] + (stop_coord[1] - prev_coord[1]) * ratio
                        coordinates.append((interp_lat, interp_lon))
                
                # ALWAYS insert exact stop coordinate (ensures visual connection)
                if not coordinates or coordinates[-1] != stop_coord:
                    coordinates.append(stop_coord)
            
            # Return to DC
            last_stop = stops_by_id[stop_ids[-1]]
            last_stop_coord = (last_stop["lat"], last_stop["lon"])
            last_leg_key = (stop_ids[-1], dc.id)
            leg = leg_map.get(last_leg_key)
            if leg:
                leg_geom = leg.get("geometry") or leg.get("polyline")
                leg_coords = normalize_geometry_to_latlon(leg_geom) if leg_geom else []
                if leg_coords and len(leg_coords) > 1:
                    # Add intermediate road path coordinates back to DC
                    for coord in leg_coords:
                        # Skip if it's the exact last stop coordinate
                        if abs(coord[0] - last_stop_coord[0]) < 1e-8 and abs(coord[1] - last_stop_coord[1]) < 1e-8:
                            continue
                        # Skip if it's the exact DC coordinate (we'll add it explicitly)
                        if abs(coord[0] - dc.lat) < 1e-8 and abs(coord[1] - dc.lon) < 1e-8:
                            continue
                        # Skip if it's a duplicate of the last coordinate
                        if coordinates and abs(coordinates[-1][0] - coord[0]) < 1e-8 and abs(coordinates[-1][1] - coord[1]) < 1e-8:
                            continue
                        coordinates.append(coord)
            
            # Add interpolated points if distance to DC is large
            if coordinates:
                prev_coord = coordinates[-1]
                dist_to_dc = ((dc.lat - prev_coord[0])**2 + (dc.lon - prev_coord[1])**2)**0.5
                if dist_to_dc > 0.0005:  # Lower threshold for better continuity
                    num_interp = min(8, max(3, int(dist_to_dc / 0.0002)))
                    for i in range(1, num_interp + 1):
                        ratio = i / (num_interp + 1)
                        interp_lat = prev_coord[0] + (dc.lat - prev_coord[0]) * ratio
                        interp_lon = prev_coord[1] + (dc.lon - prev_coord[1]) * ratio
                        coordinates.append((interp_lat, interp_lon))
            
            # ALWAYS end at DC (ensures route returns to depot)
            if not coordinates or coordinates[-1] != (dc.lat, dc.lon):
                coordinates.append((dc.lat, dc.lon))
        
        # Ensure route ends at DC
        if coordinates:
            last_coord = coordinates[-1]
            dist_to_dc = ((last_coord[0] - dc.lat)**2 + (last_coord[1] - dc.lon)**2)**0.5
            if dist_to_dc > 0.001:  # Not close to DC
                coordinates.append((dc.lat, dc.lon))
    
    # Fallback: build from stop sequence (straight lines)
    if not coordinates and "ordered_stop_ids" in route:
        stop_ids = route["ordered_stop_ids"]
        logger.warning("FALLBACK straight line - no geometry available")
        
        # Add DC start
        coordinates.append((dc.lat, dc.lon))
        
        # Add stops
        for stop_id in stop_ids:
            if stop_id in stops_by_id:
                stop = stops_by_id[stop_id]
                coordinates.append((stop["lat"], stop["lon"]))
        
        # Add DC end
        coordinates.append((dc.lat, dc.lon))
    
    # Debug: For first vehicle of first depot, assert geometry quality
    if coordinates and len(coordinates) < 20:
        logger.warning(f"Route has only {len(coordinates)} points - may not be using OSRM geometry")
    
    return coordinates


def save_route_map_html(
    baseline_solution: Dict[str, Any],
    improved_solution: Optional[Dict[str, Any]],
    dcs: List[Any],
    stops_by_id: Dict[str, Dict[str, Any]],
    out_html: str,
    tiles: str = "OpenStreetMap",
    toggle_mode: Literal["radio", "checkbox"] = "radio"
) -> None:
    """
    Generate interactive HTML map with baseline and improved routes.
    
    Args:
        baseline_solution: Baseline solution with routes_by_dc
        improved_solution: Optional improved solution
        dcs: List of DC configs
        stops_by_id: Dict mapping stop ID to stop data
        out_html: Output HTML file path
        tiles: Map tile provider (default: "OpenStreetMap")
        toggle_mode: "radio" for exclusive toggle, "checkbox" for independent layers
    """
    if folium is None:
        raise ImportError("folium is required. Install with: pip install folium")
    
    # Collect all coordinates for centering
    all_coords = []
    for dc in dcs:
        all_coords.append((dc.lat, dc.lon))
    for stop in stops_by_id.values():
        all_coords.append((stop["lat"], stop["lon"]))
    
    if not all_coords:
        logger.warning("No coordinates found for map centering")
        return
    
    # Calculate center
    center_lat = sum(c[0] for c in all_coords) / len(all_coords)
    center_lon = sum(c[1] for c in all_coords) / len(all_coords)
    
    # Create map with tiles
    # Use OpenStreetMap as default if tiles fail, or try without tiles
    try:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=tiles
        )
    except Exception as e:
        logger.warning(f"Failed to create map with tiles '{tiles}': {e}, using OpenStreetMap")
        try:
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles="OpenStreetMap"
            )
        except Exception as e2:
            logger.warning(f"Failed to create map with OpenStreetMap: {e2}, using no tiles")
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles=None
            )
    
    # Create 4 FeatureGroups: stops, baseline routes, improved routes, both routes
    fg_stops = folium.FeatureGroup(name="Stops", overlay=True, show=True)
    fg_base = folium.FeatureGroup(name="Routes: Baseline", overlay=True, show=True)
    fg_impr = folium.FeatureGroup(name="Routes: Improved", overlay=True, show=False)
    fg_both = folium.FeatureGroup(name="Routes: Both", overlay=True, show=False)
    
    # Add DC markers with different colors - BRIGHT and VISIBLE
    # Use standard Folium colors for icons, hex for lines
    dc_icon_colors = ["red", "blue", "green", "purple", "orange", "darkred", "darkblue", "darkgreen", "cadetblue", "pink"]
    dc_line_colors = ["#FF0000", "#0066FF", "#00CC00", "#CC00CC", "#FF6600", "#CC0000", "#0000CC", "#00AA00", "#AA00AA", "#FF3300"]
    for dc_idx, dc in enumerate(dcs):
        dc_icon_color = dc_icon_colors[dc_idx % len(dc_icon_colors)]
        folium.Marker(
            [dc.lat, dc.lon],
            popup=f"DC: {dc.id}",
            icon=folium.Icon(color=dc_icon_color, icon="warehouse", prefix="fa")
        ).add_to(fg_stops)
    
    # Add stop markers - ensure ALL stops are visible
    stop_count = 0
    for stop_id, stop in stops_by_id.items():
        # Validate stop has coordinates
        if "lat" not in stop or "lon" not in stop:
            logger.warning(f"Skipping stop {stop_id}: missing coordinates")
            continue
        if not isinstance(stop["lat"], (int, float)) or not isinstance(stop["lon"], (int, float)):
            logger.warning(f"Skipping stop {stop_id}: invalid coordinate types")
            continue
        
        folium.CircleMarker(
            [stop["lat"], stop["lon"]],
            radius=6,  # Larger for better visibility
            popup=f"Stop: {stop_id} (demand: {stop.get('demand', 'N/A')})",
            color="#0000FF",  # Bright blue
            fill=True,
            fillColor="#0000FF",
            fillOpacity=0.9,  # More opaque
            weight=2  # Thicker border
        ).add_to(fg_stops)
        stop_count += 1
    
    logger.info(f"Added {stop_count} stop markers to map (expected {len(stops_by_id)})")
    
    # Color palette for DCs - BRIGHT and VISIBLE colors (use line colors)
    # Vehicle colors - use darker shades of DC colors for multiple vehicles
    vehicle_colors = ["#FF3333", "#3366FF", "#33CC33", "#CC33CC", "#FF9933"]
    
    # Process baseline routes - different color per DC
    baseline_routes_by_dc = baseline_solution.get("routes_by_dc", {})
    baseline_route_count = sum(len(routes) for routes in baseline_routes_by_dc.values())
    logger.info(f"Baseline solution: {baseline_route_count} routes across {len(baseline_routes_by_dc)} depots")
    
    for dc_idx, dc in enumerate(dcs):
        dc_color = dc_line_colors[dc_idx % len(dc_line_colors)]
        dc_routes = baseline_routes_by_dc.get(dc.id, [])
        logger.info(f"  DC {dc.id}: {len(dc_routes)} baseline routes")
        
        for route_idx, route in enumerate(dc_routes):
            # Use DC color as base, with slight variation for multiple vehicles
            if len(dc_routes) > 1:
                # Multiple vehicles: use vehicle color with DC color hint
                color = vehicle_colors[route_idx % len(vehicle_colors)]
            else:
                color = dc_color
            
            stop_ids = route.get("ordered_stop_ids", [])
            route_id = f"{dc.id}_{route.get('vehicle_id', f'V{route_idx}')}"
            logger.debug(f"  Baseline route {route_id}: {len(stop_ids)} stops: {stop_ids[:5]}...")
            
            coords = extract_route_coordinates(route, stops_by_id, dc)
            if coords:
                # Add to baseline layer
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=5,  # Thicker lines
                    opacity=1.0,  # Fully opaque
                    popup=f"Baseline {dc.id} - {route.get('vehicle_id', 'V?')} ({len(stop_ids)} stops)"
                ).add_to(fg_base)
                
                # Also add to "both" layer with baseline styling
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=4,
                    opacity=0.7,
                    popup=f"Baseline {dc.id} - {route.get('vehicle_id', 'V?')} ({len(stop_ids)} stops)"
                ).add_to(fg_both)
                
                logger.debug(f"Added baseline route {route_id}: {len(stop_ids)} stops, {len(coords)} coordinates")
            else:
                logger.warning(f"Failed to extract coordinates for baseline route {route_id}")
    
    # Process improved routes - same DC colors but dashed
    # Recalculate geometry using OSRM for improved routes
    improved_routes_by_dc = {}
    if improved_solution:
        improved_routes_by_dc = improved_solution.get("routes_by_dc", {})
        improved_route_count = sum(len(routes) for routes in improved_routes_by_dc.values())
        logger.info(f"Improved solution: {improved_route_count} routes across {len(improved_routes_by_dc)} depots")
        
        # Create OSRM cache for geometry calculation
        cache = None
        if OSRM_AVAILABLE:
            cache = iNaviCache(approx_mode=False)
            logger.info("Recalculating OSRM geometry for improved routes...")
        
        # Compare with baseline
        for dc_id in set(list(baseline_routes_by_dc.keys()) + list(improved_routes_by_dc.keys())):
            baseline_routes = baseline_routes_by_dc.get(dc_id, [])
            improved_routes = improved_routes_by_dc.get(dc_id, [])
            
            baseline_stops = set()
            for route in baseline_routes:
                baseline_stops.update(route.get("ordered_stop_ids", []))
            
            improved_stops = set()
            for route in improved_routes:
                improved_stops.update(route.get("ordered_stop_ids", []))
            
            if baseline_stops != improved_stops:
                logger.info(f"  DC {dc_id}: Routes DIFFER - baseline stops: {sorted(baseline_stops)}, improved stops: {sorted(improved_stops)}")
            else:
                logger.info(f"  DC {dc_id}: Routes SAME - {len(baseline_stops)} stops")
        
        for dc_idx, dc in enumerate(dcs):
            dc_color = dc_line_colors[dc_idx % len(dc_line_colors)]
            dc_routes = improved_routes_by_dc.get(dc.id, [])
            logger.info(f"  DC {dc.id}: {len(dc_routes)} improved routes")
            
            for route_idx, route in enumerate(dc_routes):
                if len(dc_routes) > 1:
                    color = vehicle_colors[route_idx % len(vehicle_colors)]
                else:
                    color = dc_color
                
                stop_ids = route.get("ordered_stop_ids", [])
                route_id = f"{dc.id}_{route.get('vehicle_id', f'V{route_idx}')}"
                logger.debug(f"  Improved route {route_id}: {len(stop_ids)} stops: {stop_ids[:5]}...")
                
                # Recalculate geometry using OSRM if not present
                has_geometry = bool(route.get("geometry"))
                if not has_geometry and OSRM_AVAILABLE and cache and stop_ids:
                    logger.info(f"  Recalculating OSRM geometry for {route_id} (no geometry in route)...")
                    try:
                        # Build geometry by concatenating leg routes
                        geometry_coords = []
                        
                        # DC -> first stop
                        if stop_ids and stop_ids[0] in stops_by_id:
                            first_stop = stops_by_id[stop_ids[0]]
                            leg = get_osrm_leg(
                                origin=(dc.lat, dc.lon),
                                dest=(first_stop["lat"], first_stop["lon"]),
                                cache=cache
                            )
                            geometry_coords.extend(leg.get("polyline", []))
                        
                        # Stop -> stop legs
                        for i in range(len(stop_ids) - 1):
                            from_id = stop_ids[i]
                            to_id = stop_ids[i + 1]
                            if from_id in stops_by_id and to_id in stops_by_id:
                                from_stop = stops_by_id[from_id]
                                to_stop = stops_by_id[to_id]
                                leg = get_osrm_leg(
                                    origin=(from_stop["lat"], from_stop["lon"]),
                                    dest=(to_stop["lat"], to_stop["lon"]),
                                    cache=cache
                                )
                                leg_poly = leg.get("polyline", [])
                                # Skip first point (duplicate of previous leg's last point)
                                if leg_poly and len(leg_poly) > 1:
                                    geometry_coords.extend(leg_poly[1:])
                                elif leg_poly:
                                    geometry_coords.extend(leg_poly)
                        
                        # Last stop -> DC
                        if stop_ids and stop_ids[-1] in stops_by_id:
                            last_stop = stops_by_id[stop_ids[-1]]
                            leg = get_osrm_leg(
                                origin=(last_stop["lat"], last_stop["lon"]),
                                dest=(dc.lat, dc.lon),
                                cache=cache
                            )
                            leg_poly = leg.get("polyline", [])
                            if leg_poly and len(leg_poly) > 1:
                                geometry_coords.extend(leg_poly[1:])
                            elif leg_poly:
                                geometry_coords.extend(leg_poly)
                        
                        if geometry_coords:
                            # Store as geometry for future use
                            route["geometry"] = geometry_coords
                            logger.info(f"  ✓ Calculated {len(geometry_coords)} OSRM geometry points for {route_id}")
                            if len(geometry_coords) > 20:
                                logger.debug(f"  Curvy route confirmed (>20 points)")
                        else:
                            logger.warning(f"  No geometry calculated for {route_id}")
                    except Exception as e:
                        logger.warning(f"  Failed to calculate geometry for {route_id}: {e}")
                
                coords = extract_route_coordinates(route, stops_by_id, dc)
                if coords:
                    # Add to improved layer (dashed)
                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=5,  # Thicker lines
                        opacity=1.0,  # Fully opaque
                        popup=f"Improved {dc.id} - {route.get('vehicle_id', 'V?')} ({len(stop_ids)} stops)",
                        dashArray="15, 10"  # More visible dashed line for improved
                    ).add_to(fg_impr)
                    
                    # Also add to "both" layer with improved styling (dashed, different opacity)
                    folium.PolyLine(
                        coords,
                        color=color,
                        weight=4,
                        opacity=0.7,
                        popup=f"Improved {dc.id} - {route.get('vehicle_id', 'V?')} ({len(stop_ids)} stops)",
                        dashArray="10, 5"  # Dashed for improved in "both" view
                    ).add_to(fg_both)
                    
                    logger.debug(f"Added improved route {route_id}: {len(stop_ids)} stops, {len(coords)} coordinates")
                else:
                    logger.warning(f"Failed to extract coordinates for improved route {route_id}")
    else:
        logger.info("No improved solution provided")
    
    # Add feature groups to map
    fg_stops.add_to(m)
    fg_base.add_to(m)
    if improved_solution:
        fg_impr.add_to(m)
        fg_both.add_to(m)
    
    # Add layer control based on toggle_mode
    if toggle_mode == "radio":
        # Try GroupedLayerControl for exclusive toggle
        try:
            from folium.plugins import GroupedLayerControl
            
            route_groups = [fg_base]
            if improved_solution:
                route_groups.append(fg_impr)
                route_groups.append(fg_both)
            
            GroupedLayerControl(
                groups={"Routes": route_groups, "Data": [fg_stops]},
                collapsed=False,
                exclusive_groups=["Routes"]
            ).add_to(m)
            logger.info("Using GroupedLayerControl for exclusive route toggle")
        except (ImportError, AttributeError):
            # Fallback: Use JS to make baseline/improved mutually exclusive
            logger.info("GroupedLayerControl not available, using JS fallback for exclusive toggle")
            folium.LayerControl(collapsed=False).add_to(m)
            
            # Inject JS to make baseline/improved/both mutually exclusive
            base_name = fg_base.get_name()
            impr_name = fg_impr.get_name() if improved_solution else None
            both_name = fg_both.get_name() if improved_solution else None
            
            js_code = f"""
            <script>
            (function() {{
                function setupExclusiveToggle() {{
                    var control = document.querySelector('.leaflet-control-layers');
                    if (!control) {{
                        setTimeout(setupExclusiveToggle, 100);
                        return;
                    }}
                    
                    // Find checkboxes by label text (more reliable than value)
                    var labels = control.querySelectorAll('label');
                    var baseCheckbox = null;
                    var imprCheckbox = null;
                    
                    var bothCheckbox = null;
                    
                    labels.forEach(function(label) {{
                        var text = label.textContent.trim();
                        var checkbox = label.querySelector('input[type="checkbox"]');
                        if (!checkbox) return;
                        
                        if (text.includes('Baseline')) {{
                            baseCheckbox = checkbox;
                        }} else if (text.includes('Improved')) {{
                            imprCheckbox = checkbox;
                        }} else if (text.includes('Both')) {{
                            bothCheckbox = checkbox;
                        }}
                    }});
                    
                    function makeExclusive(checkboxes) {{
                        checkboxes.forEach(function(cb) {{
                            if (!cb) return;
                            cb.addEventListener('change', function() {{
                                if (this.checked) {{
                                    checkboxes.forEach(function(other) {{
                                        if (other && other !== this && other.checked) {{
                                            other.checked = false;
                                            other.dispatchEvent(new Event('change'));
                                        }}
                                    }}.bind(this));
                                }}
                            }});
                        }});
                    }}
                    
                    var routeCheckboxes = [baseCheckbox, imprCheckbox, bothCheckbox].filter(function(cb) {{ return cb !== null; }});
                    if (routeCheckboxes.length > 1) {{
                        makeExclusive(routeCheckboxes);
                    }}
                }}
                
                // Wait for map to be fully loaded
                if (document.readyState === 'loading') {{
                    document.addEventListener('DOMContentLoaded', setupExclusiveToggle);
                }} else {{
                    setTimeout(setupExclusiveToggle, 300);
                }}
            }})();
            </script>
            """
            
            from folium import Element
            m.get_root().html.add_child(Element(js_code))
    else:
        # Checkbox mode: allow both layers independently
        folium.LayerControl(collapsed=False).add_to(m)
        logger.info("Using checkbox mode - both layers can be visible simultaneously")
    
    # Save map
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    
    logger.info(f"Map saved to: {out_path.absolute()}")

