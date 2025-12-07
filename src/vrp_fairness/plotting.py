"""
Visualization module for VRP fairness solutions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np

from .config import DCConfig
from .metrics import calculate_route_metrics, SolutionMetrics

logger = logging.getLogger(__name__)


def extract_route_coordinates(
    route: Dict[str, Any],
    stops_dict: Dict[str, Dict[str, Any]],
    dc: DCConfig,
    use_polylines: bool = True
) -> Tuple[List[float], List[float]]:
    """
    Extract coordinates for a route's stop sequence.
    If polylines are available, use them; otherwise use straight lines.
    
    Args:
        route: Route dict with 'ordered_stop_ids' and optionally 'legs' with polylines
        stops_dict: Dictionary mapping stop_id to stop data
        dc: DC configuration (for depot coordinates)
        use_polylines: If True, use polylines from route legs if available
    
    Returns:
        (lons, lats) as lists of coordinates
    """
    # Check if polylines are available
    if use_polylines and "legs" in route and route["legs"]:
        # Use polylines from route legs
        lons = []
        lats = []
        
        for leg in route["legs"]:
            if "polyline" in leg and leg["polyline"]:
                # leg["polyline"] is a list of (lat, lon) tuples
                for lat, lon in leg["polyline"]:
                    lons.append(lon)
                    lats.append(lat)
            else:
                # Fallback: use from/to coordinates
                from_id = leg.get("from")
                to_id = leg.get("to")
                
                if from_id == dc.id:
                    lons.append(dc.lon)
                    lats.append(dc.lat)
                elif from_id in stops_dict:
                    lons.append(stops_dict[from_id]["lon"])
                    lats.append(stops_dict[from_id]["lat"])
                
                if to_id == dc.id:
                    lons.append(dc.lon)
                    lats.append(dc.lat)
                elif to_id in stops_dict:
                    lons.append(stops_dict[to_id]["lon"])
                    lats.append(stops_dict[to_id]["lat"])
        
        # Remove duplicates at segment boundaries
        if len(lons) > 1:
            cleaned_lons = [lons[0]]
            cleaned_lats = [lats[0]]
            for i in range(1, len(lons)):
                if abs(lons[i] - cleaned_lons[-1]) > 1e-6 or abs(lats[i] - cleaned_lats[-1]) > 1e-6:
                    cleaned_lons.append(lons[i])
                    cleaned_lats.append(lats[i])
            return cleaned_lons, cleaned_lats
        
        return lons, lats
    
    # Fallback: use stop sequence (straight lines) - only if polylines completely unavailable
    # NOTE: This should rarely happen if get_leg_route() is used properly
    logger.warning(f"No polylines available for route {route.get('vehicle_id', 'unknown')}, using straight lines")
    lons = []
    lats = []
    
    stop_ids = route.get("ordered_stop_ids", route.get("stops", []))
    for stop_id in stop_ids:
        if stop_id == dc.id or stop_id.startswith("DC"):
            # Depot
            lons.append(dc.lon)
            lats.append(dc.lat)
        elif stop_id in stops_dict:
            # Regular stop
            stop_data = stops_dict[stop_id]
            lons.append(stop_data["lon"])
            lats.append(stop_data["lat"])
    
    return lons, lats


def extract_waiting_times(
    routes: List[Dict[str, Any]],
    stops_dict: Dict[str, Dict[str, Any]],
    time_matrix: Dict[tuple, int],
    dc_id: str
) -> List[float]:
    """
    Extract waiting times for all stops in routes.
    
    Args:
        routes: List of route dictionaries
        stops_dict: Dictionary mapping stop_id to stop data
        time_matrix: Dictionary mapping (stop1_id, stop2_id) to travel time
        dc_id: DC ID (depot ID)
    
    Returns:
        List of waiting times in seconds
    """
    waiting_times = []
    
    for route in routes:
        route_metrics = calculate_route_metrics(
            route, stops_dict, time_matrix, depot_id=dc_id
        )
        waiting_times.extend(route_metrics.waiting_times.values())
    
    return waiting_times


def plot_routes_baseline_vs_improved(
    baseline_solution: Dict[str, Any],
    improved_solution: Optional[Dict[str, Any]],
    dcs: List[DCConfig],
    city_name: str,
    output_path: str,
    plot_format: str = "png"
) -> None:
    """
    Plot baseline vs improved routes for comparison.
    
    Args:
        baseline_solution: Baseline solution dictionary
        improved_solution: Improved solution dictionary (None if no improvement)
        dcs: List of DC configurations
        city_name: City name for title
        output_path: Output file path
        plot_format: Plot format ("png" or "pdf")
    """
    fig, axes = plt.subplots(1, len(dcs), figsize=(12 * len(dcs), 10))
    if len(dcs) == 1:
        axes = [axes]
    
    for dc_idx, dc in enumerate(dcs):
        ax = axes[dc_idx]
        
        # Get routes for this DC
        baseline_routes = baseline_solution["routes_by_dc"].get(dc.id, [])
        improved_routes = (
            improved_solution["routes_by_dc"].get(dc.id, [])
            if improved_solution else []
        )
        
        stops_dict = baseline_solution["stops_dict"]
        
        # Plot DC
        ax.scatter(
            [dc.lon], [dc.lat],
            c='red', marker='*', s=500, zorder=10,
            label='DC', edgecolors='black', linewidths=2
        )
        
        # Plot baseline routes
        colors_baseline = plt.cm.Set2(np.linspace(0, 1, len(baseline_routes)))
        for route_idx, route in enumerate(baseline_routes):
            lons, lats = extract_route_coordinates(route, stops_dict, dc, use_polylines=True)
            
            if len(lons) > 1:
                ax.plot(
                    lons, lats,
                    '--', alpha=0.6, linewidth=1.5,
                    color=colors_baseline[route_idx],
                    label=f'Baseline V{route.get("vehicle_id", route_idx+1)}' if route_idx < 3 else None
                )
                # Plot stop markers (exclude DC at start/end if present)
                stop_ids = route.get("ordered_stop_ids", route.get("stops", []))
                stop_coords = []
                for stop_id in stop_ids[1:-1] if len(stop_ids) > 2 else stop_ids[1:]:
                    if stop_id in stops_dict:
                        stop_coords.append((stops_dict[stop_id]["lon"], stops_dict[stop_id]["lat"]))
                if stop_coords:
                    stop_lons, stop_lats = zip(*stop_coords)
                    ax.scatter(
                        stop_lons, stop_lats,
                        color=colors_baseline[route_idx], marker='o',
                        s=50, alpha=0.6, zorder=5
                    )
        
        # Plot improved routes
        if improved_routes:
            colors_improved = plt.cm.Dark2(np.linspace(0, 1, len(improved_routes)))
            for route_idx, route in enumerate(improved_routes):
                lons, lats = extract_route_coordinates(route, stops_dict, dc, use_polylines=True)
                
                if len(lons) > 1:
                    ax.plot(
                        lons, lats,
                        '-', alpha=0.8, linewidth=2.5,
                        color=colors_improved[route_idx],
                        label=f'Improved V{route.get("vehicle_id", route_idx+1)}' if route_idx < 3 else None
                    )
                    # Plot stop markers (exclude DC at start/end if present)
                    stop_ids = route.get("ordered_stop_ids", route.get("stops", []))
                    stop_coords = []
                    for stop_id in stop_ids[1:-1] if len(stop_ids) > 2 else stop_ids[1:]:
                        if stop_id in stops_dict:
                            stop_coords.append((stops_dict[stop_id]["lon"], stops_dict[stop_id]["lat"]))
                    if stop_coords:
                        stop_lons, stop_lats = zip(*stop_coords)
                        ax.scatter(
                            stop_lons, stop_lats,
                            color=colors_improved[route_idx], marker='s',
                            s=80, alpha=0.8, zorder=6, edgecolors='black', linewidths=0.5
                        )
        else:
            # No improvement case
            ax.text(
                0.5, 0.95, 'No improvement found',
                transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=12
            )
        
        # Formatting
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'{dc.name or dc.id} - {city_name.title()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9, ncol=2)
        
        # Set equal aspect ratio for better visualization
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(
        f'Route Comparison: Baseline vs Improved ({city_name.title()})',
        fontsize=16, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format=plot_format, dpi=150, bbox_inches='tight')
    plt.close()


def plot_waiting_time_histograms(
    baseline_waits: List[float],
    improved_waits: Optional[List[float]],
    city_name: str,
    output_path: str,
    plot_format: str = "png"
) -> None:
    """
    Plot side-by-side histograms of waiting times.
    
    Args:
        baseline_waits: List of baseline waiting times
        improved_waits: List of improved waiting times (None if no improvement)
        city_name: City name for title
        output_path: Output file path
        plot_format: Plot format ("png" or "pdf")
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if not baseline_waits:
        ax.text(0.5, 0.5, 'No waiting time data available',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Waiting Time Distribution ({city_name.title()})', fontsize=14)
        plt.savefig(output_path, format=plot_format, dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Determine bins
    all_waits = baseline_waits + (improved_waits or [])
    max_wait = max(all_waits) if all_waits else 1.0
    bins = np.linspace(0, max_wait * 1.1, min(30, max(10, len(baseline_waits) // 3)))
    
    # Plot histograms
    if improved_waits:
        ax.hist(
            baseline_waits, bins=bins,
            alpha=0.6, label='Baseline',
            color='skyblue', edgecolor='black', linewidth=1
        )
        ax.hist(
            improved_waits, bins=bins,
            alpha=0.6, label='Improved',
            color='coral', edgecolor='black', linewidth=1
        )
    else:
        ax.hist(
            baseline_waits, bins=bins,
            alpha=0.7, label='Baseline',
            color='skyblue', edgecolor='black', linewidth=1
        )
        ax.text(
            0.5, 0.95, 'No improvement data available',
            transform=ax.transAxes,
            ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=12
        )
    
    # Statistics text
    if baseline_waits:
        baseline_mean = np.mean(baseline_waits)
        baseline_max = np.max(baseline_waits)
        stats_text = f'Baseline: Mean={baseline_mean:.1f}s, Max={baseline_max:.1f}s'
        
        if improved_waits:
            improved_mean = np.mean(improved_waits)
            improved_max = np.max(improved_waits)
            improvement_pct = ((baseline_max - improved_max) / baseline_max * 100) if baseline_max > 0 else 0
            stats_text += f'\nImproved: Mean={improved_mean:.1f}s, Max={improved_max:.1f}s'
            stats_text += f'\nMax Reduction: {improvement_pct:.1f}%'
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10, family='monospace'
        )
    
    # Formatting
    ax.set_xlabel('Waiting Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Waiting Time Distribution: Baseline vs Improved ({city_name.title()})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format=plot_format, dpi=150, bbox_inches='tight')
    plt.close()

