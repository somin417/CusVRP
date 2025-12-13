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
            # Quick validation: check if geometry starts at DC and ends at DC (full round trip)
            stop_ids = route.get("ordered_stop_ids", [])
            if stop_ids and stops_by_id:
                first_stop = stops_by_id.get(stop_ids[0])
                last_stop = stops_by_id.get(stop_ids[-1])
                if first_stop and last_stop:
                    # Check if geometry starts at DC and ends at DC (round trip)
                    geom_start = normalized[0]
                    geom_end = normalized[-1]
                    start_to_dc = ((geom_start[0] - dc.lat)**2 + (geom_start[1] - dc.lon)**2)**0.5
                    end_to_dc = ((geom_end[0] - dc.lat)**2 + (geom_end[1] - dc.lon)**2)**0.5
                    
                    # VROOM geometry should start and end at DC (round trip: DC -> stops -> DC)
                    # If geometry doesn't start/end at DC, it may be incomplete
                    if start_to_dc > 0.01 or end_to_dc > 0.01:  # ~1km threshold
                        logger.warning(f"Geometry doesn't start/end at DC (start_dist={start_to_dc:.4f}, end_dist={end_to_dc:.4f}), recalculating from stop order")
                        route_geometry = None  # Force fallback to stop order
                    else:
                        logger.info(f"DRAWING GEOMETRY route-level polyline points={len(normalized)} (includes DC->stops->DC)")
                        return normalized
            
            if normalized and len(normalized) > 20:
                logger.info(f"DRAWING GEOMETRY route-level polyline points={len(normalized)} (includes DC->stops->DC)")
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
                    # Skip first point if it's duplicate of previous leg's last point
                    if all_leg_coords and len(normalized) > 1:
                        last_coord = all_leg_coords[-1]
                        first_coord = normalized[0]
                        if abs(last_coord[0] - first_coord[0]) < 1e-6 and abs(last_coord[1] - first_coord[1]) < 1e-6:
                            all_leg_coords.extend(normalized[1:])
                        else:
                            all_leg_coords.extend(normalized)
                    else:
                        all_leg_coords.extend(normalized)
        
        # Verify that legs include return to DC (last leg should be last_stop -> DC)
        if all_leg_coords and stop_ids:
            # Check if last leg ends at DC
            last_leg = route["legs"][-1] if route["legs"] else None
            if last_leg:
                last_leg_to = last_leg.get("to", "")
                # Check if last leg goes to depot (could be depot name or DC id)
                ends_at_dc = (last_leg_to == dc.id or 
                             last_leg_to == dc.get("name", "") or
                             last_leg_to in ["depot", "Depot", "DC"] or
                             (all_leg_coords and 
                              abs(all_leg_coords[-1][0] - dc.lat) < 0.01 and 
                              abs(all_leg_coords[-1][1] - dc.lon) < 0.01))
                
                if not ends_at_dc:
                    logger.warning(f"Per-leg geometry doesn't end at DC (last leg to={last_leg_to}), will add DC return in fallback")
                    # Don't return early, fall through to add DC return
                else:
                    if total_points > 20:
                        logger.info(f"DRAWING GEOMETRY per-leg polyline points={total_points} (includes DC->stops->DC)")
                        return all_leg_coords
        
        if total_points > 20:
            logger.info(f"DRAWING GEOMETRY per-leg polyline points={total_points} (includes DC->stops->DC)")
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
                
                # Add exact stop coordinate (OSRM polyline already provides smooth path)
                if not coordinates or abs(coordinates[-1][0] - stop_coord[0]) > 1e-6 or abs(coordinates[-1][1] - stop_coord[1]) > 1e-6:
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
            
            # End at DC (OSRM polyline already provides smooth path)
            if coordinates and (abs(coordinates[-1][0] - dc.lat) > 1e-6 or abs(coordinates[-1][1] - dc.lon) > 1e-6):
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
        # Use GroupedLayerControl for exclusive toggle
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
            # Fallback: regular layer control (not exclusive)
            logger.warning("GroupedLayerControl not available, using regular LayerControl (not exclusive)")
            folium.LayerControl(collapsed=False).add_to(m)
    else:
        # Checkbox mode: allow both layers independently
        folium.LayerControl(collapsed=False).add_to(m)
        logger.info("Using checkbox mode - both layers can be visible simultaneously")
    
    # Save map
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    
    logger.info(f"Map saved to: {out_path.absolute()}")


def _get_vehicle_color(depot_color: str, vehicle_idx: int, total_vehicles: int) -> str:
    """
    Get vehicle color as slight variation of depot color.
    
    Args:
        depot_color: Base depot color (hex, e.g., "#FF0000")
        vehicle_idx: Vehicle index within depot (0-based)
        total_vehicles: Total vehicles in this depot
    
    Returns:
        Hex color string
    """
    if total_vehicles == 1:
        return depot_color
    
    # Parse RGB from hex
    r = int(depot_color[1:3], 16)
    g = int(depot_color[3:5], 16)
    b = int(depot_color[5:7], 16)
    
    # Create variations: darker, lighter, or slightly shifted
    variations = []
    for i in range(total_vehicles):
        if i == 0:
            variations.append(depot_color)  # Base color
        else:
            # Alternate between darker and lighter
            factor = 0.85 + (i % 2) * 0.15  # 0.85 (darker) or 1.0 (base) or 1.15 (lighter)
            if i % 2 == 1:
                factor = 0.75  # Darker variant
            else:
                factor = 1.15  # Lighter variant
            
            new_r = max(0, min(255, int(r * factor)))
            new_g = max(0, min(255, int(g * factor)))
            new_b = max(0, min(255, int(b * factor)))
            variations.append(f"#{new_r:02X}{new_g:02X}{new_b:02X}")
    
    return variations[vehicle_idx % len(variations)]


def save_multi_solution_map_html(
    solutions: Dict[str, Dict[str, Any]],
    dcs: List[Any],
    stops_by_id: Dict[str, Dict[str, Any]],
    out_html: str,
    tiles: str = "OpenStreetMap"
) -> None:
    """
    Generate interactive HTML map with multiple solutions (flexible number).
    Each solution gets a radio button layer. Routes organized by depot, vehicles have color variations.
    
    Args:
        solutions: Dict mapping solution name to solution dict, e.g.:
            {"baseline": {...}, "local": {...}, "improved": {...}, "cts": {...}}
        dcs: List of DC configs (with .id, .lat, .lon)
        stops_by_id: Dict mapping stop ID to stop data
        out_html: Output HTML file path
        tiles: Map tile provider (default: "OpenStreetMap")
    """
    if folium is None:
        raise ImportError("folium is required. Install with: pip install folium")
    
    if not solutions:
        logger.warning("No solutions provided for map generation")
        return
    
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
    
    # Create map
    try:
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles=tiles
        )
    except Exception as e:
        logger.warning(f"Failed to create map with tiles '{tiles}': {e}, using OpenStreetMap")
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles="OpenStreetMap"
        )
    
    # Depot colors (distinct per depot)
    depot_colors = ["#FF0000", "#0066FF", "#00CC00", "#CC00CC", "#FF6600", "#CC0000", "#0000CC", "#00AA00"]
    depot_icon_colors = ["red", "blue", "green", "purple", "orange", "darkred", "darkblue", "darkgreen"]
    
    # Create data layer (stops and depots - always visible)
    fg_data = folium.FeatureGroup(name="Data", overlay=True, show=True)
    
    # Add depot markers
    for dc_idx, dc in enumerate(dcs):
        icon_color = depot_icon_colors[dc_idx % len(depot_icon_colors)]
        folium.Marker(
            [dc.lat, dc.lon],
            popup=f"DC: {dc.id}",
            icon=folium.Icon(color=icon_color, icon="warehouse", prefix="fa")
        ).add_to(fg_data)
    
    # Add stop markers
    for stop_id, stop in stops_by_id.items():
        if "lat" not in stop or "lon" not in stop:
            continue
        folium.CircleMarker(
            [stop["lat"], stop["lon"]],
            radius=5,
            popup=f"Stop: {stop_id} (demand: {stop.get('demand', 'N/A')})",
            color="#0000FF",
            fill=True,
            fillColor="#0000FF",
            fillOpacity=0.8,
            weight=2
        ).add_to(fg_data)
    
    fg_data.add_to(m)
    
    # Create OSRM cache for geometry calculation (if needed)
    cache = None
    if OSRM_AVAILABLE:
        cache = iNaviCache(approx_mode=False)
        logger.info("OSRM cache initialized for geometry calculation")
    
    # Create one FeatureGroup per solution (radio buttons)
    solution_groups = []
    solution_names = list(solutions.keys())
    
    for sol_idx, (sol_name, solution) in enumerate(solutions.items()):
        fg = folium.FeatureGroup(
            name=sol_name.title(),
            overlay=False,
            control=True,
            show=(sol_idx == 0)  # Show first solution by default
        )
        
        routes_by_dc = solution.get("routes_by_dc", {})
        
        # Process each depot
        for dc_idx, dc in enumerate(dcs):
            dc_color = depot_colors[dc_idx % len(depot_colors)]
            dc_routes = routes_by_dc.get(dc.id, [])
            
            # Process each vehicle route
            for route_idx, route in enumerate(dc_routes):
                # Get vehicle color (variation of depot color)
                vehicle_color = _get_vehicle_color(dc_color, route_idx, len(dc_routes))
                
                stop_ids = route.get("ordered_stop_ids", [])
                route_id = f"{dc.id}_{route.get('vehicle_id', f'V{route_idx}')}"
                
                # Recalculate geometry using OSRM if not present
                has_geometry = bool(route.get("geometry"))
                if not has_geometry and OSRM_AVAILABLE and cache and stop_ids:
                    logger.info(f"  Recalculating OSRM geometry for {sol_name} {route_id} (no geometry in route)...")
                    try:
                        from .osrm_provider import get_osrm_leg
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
                            leg_poly = leg.get("polyline", [])
                            if leg_poly:
                                geometry_coords.extend(leg_poly)
                        
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
                        else:
                            logger.warning(f"  No geometry calculated for {route_id}")
                    except Exception as e:
                        logger.warning(f"  Failed to calculate geometry for {route_id}: {e}")
                
                # Extract coordinates using OSRM/iNavi polylines
                coords = extract_route_coordinates(route, stops_by_id, dc)
                
                if coords:
                    folium.PolyLine(
                        coords,
                        color=vehicle_color,
                        weight=4,
                        opacity=0.65,  # Reduced opacity for better visibility
                        popup=f"{sol_name.title()} - {dc.id} - {route.get('vehicle_id', 'V?')} ({len(stop_ids)} stops)",
                        tooltip=f"{sol_name} {dc.id} {route.get('vehicle_id', 'V?')}"
                    ).add_to(fg)
        
        fg.add_to(m)
        solution_groups.append(fg)
    
    # Add GroupedLayerControl for radio buttons
    try:
        from folium.plugins import GroupedLayerControl
        GroupedLayerControl(
            groups={
                "Solutions": solution_groups,
                "Data": [fg_data]
            },
            collapsed=False,
            exclusive_groups=["Solutions"]
        ).add_to(m)
        logger.info(f"Created map with {len(solution_groups)} solution layers using GroupedLayerControl")
    except (ImportError, AttributeError):
        # Fallback to regular LayerControl
        folium.LayerControl(collapsed=False).add_to(m)
        logger.warning("GroupedLayerControl not available, using regular LayerControl (not exclusive)")
    
    # Save map
    out_path = Path(out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    logger.info(f"Multi-solution map saved to: {out_path.absolute()}")

