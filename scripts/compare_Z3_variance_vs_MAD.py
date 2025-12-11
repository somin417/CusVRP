#!/usr/bin/env python3
"""
Compare Z3 computation: Variance vs MAD (Mean Absolute Deviation)
Runs the same experiment with both metrics and compares results.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.vrp_fairness.data import load_stops_from_gpkg
from src.vrp_fairness.assignment import assign_stops_to_depots, create_osrm_time_provider
from src.vrp_fairness.vroom_vrp import solve_multi_depot
from src.vrp_fairness.proposed_algorithm import proposed_algorithm
from src.vrp_fairness.objectives import (
    compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_Z3_MAD, compute_combined_Z
)
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_Z3_comparison_histograms(
    baseline_waits: List[float],
    variance_waits: List[float],
    mad_waits: List[float],
    city_name: str,
    output_path: Path,
    baseline_weights: List[float] = None,
    variance_weights: List[float] = None,
    mad_weights: List[float] = None,
) -> None:
    """
    Plot waiting time histograms comparing baseline, variance-improved, and MAD-improved.
    
    Args:
        baseline_waits: Baseline raw waiting times (seconds)
        variance_waits: Variance-improved raw waiting times (seconds)
        mad_waits: MAD-improved raw waiting times (seconds)
        city_name: City name for title
        output_path: Output file path (without extension)
        baseline_weights: Household weights for baseline (for frequency weighting)
        variance_weights: Household weights for variance (for frequency weighting)
        mad_weights: Household weights for MAD (for frequency weighting)
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    if not baseline_waits:
        ax.text(0.5, 0.5, 'No waiting time data available',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Waiting Time Distribution ({city_name.title()})', fontsize=14)
        plt.savefig(str(output_path) + ".png", format="png", dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Determine bins (based on raw waiting times)
    all_waits = baseline_waits + variance_waits + mad_waits
    max_wait = max(all_waits) if all_waits else 1.0
    bins = np.linspace(0, max_wait * 1.05, min(60, max(20, len(baseline_waits) // 3)))
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
    centers = bins[:-1] + bin_width / 2
    
    # Use weights for frequency weighting (like compare_waiting_and_scores.py)
    # x-axis: raw waiting time, y-axis: households-weighted frequency
    base_counts, _ = np.histogram(baseline_waits, bins=bins, weights=baseline_weights)
    var_counts, _ = np.histogram(variance_waits, bins=bins, weights=variance_weights)
    mad_counts, _ = np.histogram(mad_waits, bins=bins, weights=mad_weights)
    
    # Side-by-side bars
    offset = bin_width * 0.25
    ax.bar(
        centers - offset,
        base_counts,
        width=bin_width * 0.3,
        alpha=0.75,
        color='#4C78A8',
        label='Baseline',
        edgecolor='white',
        linewidth=0.7
    )
    ax.bar(
        centers,
        var_counts,
        width=bin_width * 0.3,
        alpha=0.75,
        color='#F58518',
        label='Variance-Improved',
        edgecolor='white',
        linewidth=0.7
    )
    ax.bar(
        centers + offset,
        mad_counts,
        width=bin_width * 0.3,
        alpha=0.75,
        color='#54A24B',
        label='MAD-Improved',
        edgecolor='white',
        linewidth=0.7
    )
    
    # Statistics text
    stats_lines = []
    if baseline_waits:
        baseline_mean = np.mean(baseline_waits)
        baseline_max = np.max(baseline_waits)
        stats_lines.append(f'Baseline: Mean={baseline_mean:.1f}s, Max={baseline_max:.1f}s')
    
    if variance_waits:
        var_mean = np.mean(variance_waits)
        var_max = np.max(variance_waits)
        var_improvement = ((baseline_max - var_max) / baseline_max * 100) if baseline_max > 0 else 0
        stats_lines.append(f'Variance: Mean={var_mean:.1f}s, Max={var_max:.1f}s ({var_improvement:+.1f}%)')
    
    if mad_waits:
        mad_mean = np.mean(mad_waits)
        mad_max = np.max(mad_waits)
        mad_improvement = ((baseline_max - mad_max) / baseline_max * 100) if baseline_max > 0 else 0
        stats_lines.append(f'MAD: Mean={mad_mean:.1f}s, Max={mad_max:.1f}s ({mad_improvement:+.1f}%)')
    
    if stats_lines:
        fig.text(
            0.01, 0.99,
            "\n".join(stats_lines),
            ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
            fontsize=10, family='monospace'
        )
    
    # Formatting
    ax.set_xlabel('Waiting Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (household-weighted)', fontsize=12)
    ax.set_title(f'Waiting Time Distribution: Z3 Variance vs MAD Comparison ({city_name.title()})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=11)
    
    # Set y-limit with small padding (like compare_waiting_and_scores.py)
    all_counts = np.concatenate([base_counts, var_counts, mad_counts])
    if all_counts.size > 0:
        max_c = all_counts.max()
        if max_c > 0:
            # Add 5% padding to max value
            ax.set_ylim(top=max_c * 1.05)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # Save PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + ".png", format="png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved waiting histogram: {output_path}.png")
    
    # Try to create interactive HTML with Plotly
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        fig_plotly = go.Figure()
        
        fig_plotly.add_trace(go.Histogram(
            x=baseline_waits,
            name='Baseline',
            opacity=0.75,
            marker_color='#4C78A8',
            nbinsx=min(60, max(20, len(baseline_waits) // 3))
        ))
        
        fig_plotly.add_trace(go.Histogram(
            x=variance_waits,
            name='Variance-Improved',
            opacity=0.75,
            marker_color='#F58518',
            nbinsx=min(60, max(20, len(variance_waits) // 3))
        ))
        
        fig_plotly.add_trace(go.Histogram(
            x=mad_waits,
            name='MAD-Improved',
            opacity=0.75,
            marker_color='#54A24B',
            nbinsx=min(60, max(20, len(mad_waits) // 3))
        ))
        
        fig_plotly.update_layout(
            title=f'Waiting Time Distribution: Z3 Variance vs MAD Comparison ({city_name.title()})',
            xaxis_title='Waiting Time (seconds)',
            yaxis_title='Frequency (household-weighted)',
            barmode='overlay',
            hovermode='x unified',
            width=1200,
            height=600
        )
        
        fig_plotly.write_html(str(output_path) + ".html")
        logger.info(f"Saved interactive histogram: {output_path}.html")
    except ImportError:
        logger.info("Plotly not available, skipping interactive HTML generation")


def run_experiment_with_Z3_mode(
    depots: list,
    vehicles: Dict[str, list],
    stops_by_depot: Dict[str, list],
    stops_by_id: Dict[str, Dict[str, Any]],
    time_provider,
    distance_provider,
    config: Dict[str, Any],
    use_MAD: bool = False
) -> Dict[str, Any]:
    """
    Run proposed algorithm with either variance or MAD for Z3.
    
    Args:
        use_MAD: If True, use compute_Z3_MAD(); else use compute_Z3()
    
    Returns:
        Dictionary with solution and metrics
    """
    # Temporarily patch compute_Z3 in proposed_algorithm
    import src.vrp_fairness.proposed_algorithm as pa_module
    import src.vrp_fairness.objectives as obj_module
    
    # Save original function
    original_compute_Z3 = obj_module.compute_Z3
    
    # Create wrapper that uses MAD if requested
    if use_MAD:
        def compute_Z3_wrapper(waiting, stops_by_id):
            return compute_Z3_MAD(waiting, stops_by_id)
    else:
        def compute_Z3_wrapper(waiting, stops_by_id):
            return original_compute_Z3(waiting, stops_by_id)
    
    # Patch the function in BOTH modules
    # NOTE: proposed_algorithm imports compute_Z3 directly, so we need to patch both
    obj_module.compute_Z3 = compute_Z3_wrapper
    pa_module.compute_Z3 = compute_Z3_wrapper
    
    try:
        # Run algorithm
        improved, debug_info = proposed_algorithm(
            depots=depots,
            vehicles=vehicles,
            stops_by_depot=stops_by_depot,
            alpha=config["alpha"],
            beta=config["beta"],
            gamma=config["gamma"],
            eps=config["eps"],
            iters=config["iters"],
            seed=config["seed"],
            time_provider=time_provider,
            distance_provider=distance_provider,
            normalize="baseline",
            use_distance_objective=config.get("use_distance", False),
            enforce_capacity=False
        )
        
        # Use best_objectives from debug_info if available (non-normalized RAW values computed by proposed_algorithm)
        # This ensures we use the exact same values that were computed internally
        best_objectives = debug_info.get("best_objectives", {})
        
        if best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]):
            # Use exact non-normalized RAW values from proposed_algorithm
            Z1_improved = best_objectives.get("Z1")
            Z2_improved = best_objectives.get("Z2")
            Z3_improved = best_objectives.get("Z3")
            
            # Validate values are not None
            if Z1_improved is None or Z2_improved is None or Z3_improved is None:
                logger.error("best_objectives contains None values! Falling back to recomputation.")
                # Force fallback by setting best_objectives to empty
                best_objectives = {}
            else:
                logger.info(f"Using RAW best objectives from proposed_algorithm debug_info:")
                logger.info(f"  Z1_improved (RAW)={Z1_improved:.2f}")
                logger.info(f"  Z2_improved (RAW)={Z2_improved:.2f}")
                logger.info(f"  Z3_improved (RAW)={Z3_improved:.2f}")
        
        # Fallback if best_objectives not available or invalid
        if not (best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]) and 
                all(best_objectives.get(k) is not None for k in ["Z1", "Z2", "Z3"])):
            # Fallback: recompute (should not happen, but handle gracefully)
            logger.warning("best_objectives not found in debug_info, recomputing...")
            baseline_solution = debug_info.get("baseline_solution", {})
            baseline_stops_dict = baseline_solution.get("stops_dict", stops_by_id)
            
            waiting_improved = compute_waiting_times(improved, baseline_stops_dict, time_provider)
            Z1_improved = compute_Z1(waiting_improved, baseline_stops_dict)
            Z2_improved = compute_Z2(improved, distance_provider, time_provider, config.get("use_distance", False))
            Z3_improved = compute_Z3(waiting_improved, baseline_stops_dict)
            logger.info(f"Recomputed RAW objectives: Z1={Z1_improved:.2f}, Z2={Z2_improved:.2f}, Z3={Z3_improved:.2f}")
        
        # Use baseline objectives from debug_info for consistency with proposed_algorithm
        # These are the RAW baseline scores that will be used as normalizers (Z1*, Z2*, Z3*)
        baseline_objectives = debug_info.get("baseline_objectives", {})
        normalizers = debug_info.get("normalizers", {})
        
        if baseline_objectives and normalizers and all(k in baseline_objectives for k in ["Z1", "Z2", "Z3"]):
            # Use exact RAW baseline values from proposed_algorithm (these were computed with the patched compute_Z3)
            Z1_baseline = baseline_objectives.get("Z1")
            Z2_baseline = baseline_objectives.get("Z2")
            Z3_baseline = baseline_objectives.get("Z3")
            # Normalizers (Z1*, Z2*, Z3*) should equal baseline values when normalize="baseline"
            Z1_star = normalizers.get("Z1_star", Z1_baseline)
            Z2_star = normalizers.get("Z2_star", Z2_baseline)
            Z3_star = normalizers.get("Z3_star", Z3_baseline)
            
            logger.info(f"Using RAW baseline objectives from proposed_algorithm debug_info:")
            logger.info(f"  Z1_baseline (RAW)={Z1_baseline:.2f}, Z2_baseline (RAW)={Z2_baseline:.2f}, Z3_baseline (RAW)={Z3_baseline:.2f}")
            logger.info(f"  Z1_star={Z1_star:.2f}, Z2_star={Z2_star:.2f}, Z3_star={Z3_star:.2f}")
        else:
            # Fallback: recompute baseline (should not happen, but handle gracefully)
            logger.warning("baseline_objectives not found in debug_info, recomputing...")
            baseline = debug_info.get("baseline_solution", {})
            
            if not baseline or not baseline.get("routes_by_dc"):
                from src.vrp_fairness.vroom_vrp import solve_multi_depot
                baseline = solve_multi_depot(
                    depots=depots,
                    vehicles_by_depot=vehicles,
                    stops_by_depot=stops_by_depot,
                    request_geometry=False
                )
                baseline["depots"] = depots
                baseline["stops_dict"] = stops_by_id
            
            if "depots" not in baseline:
                baseline["depots"] = depots
            if "stops_dict" not in baseline:
                baseline["stops_dict"] = stops_by_id
            
            # Use baseline_solution's stops_dict for consistency with proposed_algorithm
            baseline_stops_dict = baseline.get("stops_dict", stops_by_id)
            waiting_baseline = compute_waiting_times(baseline, baseline_stops_dict, time_provider)
            Z1_baseline = compute_Z1(waiting_baseline, baseline_stops_dict)
            Z2_baseline = compute_Z2(baseline, distance_provider, time_provider, config.get("use_distance", False))
            
            # Use patched compute_Z3 (which is already patched in the wrapper)
            Z3_baseline = compute_Z3(waiting_baseline, baseline_stops_dict)
            
            Z1_star, Z2_star, Z3_star = Z1_baseline, Z2_baseline, Z3_baseline
        
        # Calculate normalized Z scores: Z = alpha*(Z1/Z1*) + beta*(Z2/Z2*) + gamma*(Z3/Z3*)
        # Baseline Z should be 1.0 (using its own values as normalizers)
        Z_baseline = compute_combined_Z(
            Z1_baseline, Z2_baseline, Z3_baseline,
            Z1_star, Z2_star, Z3_star,
            config["alpha"], config["beta"], config["gamma"]
        )
        
        # Improved Z - use same normalizers (baseline scores) as baseline
        Z_improved = compute_combined_Z(
            Z1_improved, Z2_improved, Z3_improved,
            Z1_star, Z2_star, Z3_star,
            config["alpha"], config["beta"], config["gamma"]
        )
        
        # Log the calculation details
        logger.info(f"Z-score calculation:")
        logger.info(f"  Baseline: Z = {config['alpha']:.2f}*({Z1_baseline:.2f}/{Z1_star:.2f}) + {config['beta']:.2f}*({Z2_baseline:.2f}/{Z2_star:.2f}) + {config['gamma']:.2f}*({Z3_baseline:.2f}/{Z3_star:.2f}) = {Z_baseline:.4f}")
        logger.info(f"  Improved: Z = {config['alpha']:.2f}*({Z1_improved:.2f}/{Z1_star:.2f}) + {config['beta']:.2f}*({Z2_improved:.2f}/{Z2_star:.2f}) + {config['gamma']:.2f}*({Z3_improved:.2f}/{Z3_star:.2f}) = {Z_improved:.4f}")
        
        # Debug: Verify baseline Z is 1.0
        if abs(Z_baseline - 1.0) > 0.001:
            logger.warning(f"Baseline Z is not 1.0! Z_baseline={Z_baseline:.6f}, expected 1.0")
            logger.warning(f"  This indicates a normalization error. Z1*={Z1_star:.2f}, Z2*={Z2_star:.2f}, Z3*={Z3_star:.2f}")
            logger.warning(f"  Z1={Z1_baseline:.2f}, Z2={Z2_baseline:.2f}, Z3={Z3_baseline:.2f}")
        
        return {
            "baseline": {
                "Z1": Z1_baseline,
                "Z2": Z2_baseline,
                "Z3": Z3_baseline,
                "Z": Z_baseline
            },
            "improved": {
                "Z1": Z1_improved,
                "Z2": Z2_improved,
                "Z3": Z3_improved,
                "Z": Z_improved
            },
            "improvement": {
                "Z1_pct": ((Z1_baseline - Z1_improved) / Z1_baseline * 100) if Z1_baseline > 0 else (0 if Z1_improved == Z1_baseline else float('inf')),
                "Z2_pct": ((Z2_baseline - Z2_improved) / Z2_baseline * 100) if Z2_baseline > 0 else (0 if Z2_improved == Z2_baseline else float('inf')),
                "Z3_pct": ((Z3_baseline - Z3_improved) / Z3_baseline * 100) if Z3_baseline > 0 else (0 if Z3_improved == Z3_baseline else float('inf')),
                "Z_pct": ((Z_baseline - Z_improved) / Z_baseline * 100) if Z_baseline > 0 else (0 if Z_improved == Z_baseline else float('inf'))
            },
            "solution": improved,
            "debug_info": debug_info
        }
    finally:
        # Restore original function in both modules
        obj_module.compute_Z3 = original_compute_Z3
        pa_module.compute_Z3 = original_compute_Z3


def main():
    print("=" * 80)
    print("Z3 Variance vs MAD Comparison Experiment")
    print("=" * 80)
    
    # Configuration
    config = {
        "seed": 42,
        "gpkg": "data/yuseong_housing_3__point.gpkg",
        "layer": "yuseong_housing_2__point",
        "sample_n": 50,
        "num_dcs": 3,  # Use 3 depots (will use predefined Daejeon depots)
        "vehicles_per_dc": 3,
        "demand_field": "A26",
        "eps": 0.10,
        "iters": 50,  # Increased for better optimization
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 0.2,
        "use_distance": False
    }
    
    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse", action="store_true", help="Reuse outputs/baseline.json and outputs/improved.json as baseline and MAD; train variance only")
    args = parser.parse_args()

    def _build_from_baseline(baseline_path: Path):
        baseline_solution = json.loads(baseline_path.read_text())
        depots = baseline_solution.get("depots", [])
        stops_by_id = baseline_solution.get("stops_dict", {})
        routes_by_dc = baseline_solution.get("routes_by_dc", {})
        # Build stops_by_depot from routes_by_dc
        stops_by_depot = {}
        for dc_id, routes in routes_by_dc.items():
            dc_stops = set()
            for r in routes:
                for sid in r.get("ordered_stop_ids", []):
                    if sid in stops_by_id:
                        dc_stops.add(sid)
            stops_by_depot[dc_id] = [
                {
                    "id": sid,
                    "name": sid,
                    "lat": stops_by_id[sid]["lat"],
                    "lon": stops_by_id[sid]["lon"],
                    "coordinates": [stops_by_id[sid]["lat"], stops_by_id[sid]["lon"]],
                    "demand": stops_by_id[sid].get("demand", 1),
                    "households": stops_by_id[sid].get("households", stops_by_id[sid].get("demand", 1)),
                    "service_time_s": stops_by_id[sid].get("service_time_s", 300),
                }
                for sid in dc_stops
            ]
        # Synthesize vehicles (one per route, fallback 3)
        vehicles = {}
        for dc_id, routes in routes_by_dc.items():
            vcount = max(len(routes), 3)
            vehicles[dc_id] = [
                {"id": f"{dc_id}_V{j+1}", "capacity": 1000}
                for j in range(vcount)
            ]
        return baseline_solution, depots, vehicles, stops_by_depot, stops_by_id

    if args.reuse:
        # Reuse stored solutions and run only variance training
        baseline_path = Path("outputs") / "baseline.json"
        mad_path = Path("outputs") / "improved.json"
        if not baseline_path.exists() or not mad_path.exists():
            raise FileNotFoundError("Need outputs/baseline.json and outputs/improved.json to reuse.")

        baseline_solution, depots, vehicles, stops_by_depot_dict, stops_by_id = _build_from_baseline(baseline_path)

        # Compute MAD results from stored improved.json
        mad_solution = json.loads(mad_path.read_text())
        cache = iNaviCache(approx_mode=False)
        time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)

        waiting_base = compute_waiting_times(baseline_solution, stops_by_id, time_provider)
        waiting_mad = compute_waiting_times(mad_solution, stops_by_id, time_provider)
        Z1_b = compute_Z1(waiting_base, stops_by_id); Z2_b = compute_Z2(baseline_solution, distance_provider, time_provider, config.get("use_distance", False)); Z3_b = compute_Z3_MAD(waiting_base, stops_by_id)
        Z1_m = compute_Z1(waiting_mad, stops_by_id); Z2_m = compute_Z2(mad_solution, distance_provider, time_provider, config.get("use_distance", False)); Z3_m = compute_Z3_MAD(waiting_mad, stops_by_id)
        Z_b = compute_combined_Z(Z1_b, Z2_b, Z3_b, Z1_b, Z2_b, Z3_b, config["alpha"], config["beta"], config["gamma"])
        Z_m = compute_combined_Z(Z1_m, Z2_m, Z3_m, Z1_b, Z2_b, Z3_b, config["alpha"], config["beta"], config["gamma"])
        results_MAD = {
            "baseline": {"Z1": Z1_b, "Z2": Z2_b, "Z3": Z3_b, "Z": Z_b},
            "improved": {"Z1": Z1_m, "Z2": Z2_m, "Z3": Z3_m, "Z": Z_m},
            "improvement": {
                "Z1_pct": ((Z1_b - Z1_m)/Z1_b*100) if Z1_b>0 else 0,
                "Z2_pct": ((Z2_b - Z2_m)/Z2_b*100) if Z2_b>0 else 0,
                "Z3_pct": ((Z3_b - Z3_m)/Z3_b*100) if Z3_b>0 else 0,
                "Z_pct": ((Z_b - Z_m)/Z_b*100) if Z_b>0 else 0,
            },
            "solution": mad_solution,
            "debug_info": {"baseline_solution": baseline_solution, "best_objectives": {"Z1": Z1_m, "Z2": Z2_m, "Z3": Z3_m}, "baseline_objectives": {"Z1": Z1_b, "Z2": Z2_b, "Z3": Z3_b}},
        }

        # Train variance only
        results_variance = run_experiment_with_Z3_mode(
            depots, vehicles, stops_by_depot_dict, stops_by_id,
            time_provider, distance_provider, config, use_MAD=False
        )
    else:
        # Original flow: build from GPKG and run both MAD and variance
        print("\n" + "-" * 80)
        print("Loading stops from GPKG...")
        stops = load_stops_from_gpkg(
            gpkg_path=config["gpkg"],
            layer=config["layer"],
            n=config["sample_n"],
            seed=config["seed"],
            demand_field=config["demand_field"]
        )
        logger.info(f"Loaded {len(stops)} stops")
        
        # Create depots - use predefined Daejeon depots when num_dcs == 3
        print("\n" + "-" * 80)
        print("Creating depots...")
        if config["num_dcs"] == 3:
            # Use predefined Daejeon depots (from run_experiment.py)
            predefined_dcs = [
                {"id": "DC_Logen", "lat": 36.3800587, "lon": 127.3777765},
                {"id": "DC_Hanjin", "lat": 36.3711833, "lon": 127.4050933},
                {"id": "DC_CJ", "lat": 36.449416, "lon": 127.4070349}
            ]
            depots = predefined_dcs
            logger.info("Using predefined Daejeon depots:")
            for depot in depots:
                logger.info(f"  {depot['id']}: {depot['lat']:.6f}, {depot['lon']:.6f}")
        else:
            # Generate random depots
            import random
            random.seed(config["seed"])
            depots = []
            for i in range(config["num_dcs"]):
                lat = 36.3 + random.random() * 0.2
                lon = 127.3 + random.random() * 0.1
                depots.append({
                    "id": f"DC{i+1}",
                    "lat": lat,
                    "lon": lon
                })
                logger.info(f"  {depots[-1]['id']}: {lat:.6f}, {lon:.6f}")
        
        # Create vehicles
        vehicles = {}
        for depot in depots:
            vehicles[depot["id"]] = [
                {"id": f"{depot['id']}_V{j+1}", "capacity": 1000}
                for j in range(config["vehicles_per_dc"])
            ]
        
        # Assign stops to depots
        print("\n" + "-" * 80)
        print("Assigning stops to depots...")
        cache = iNaviCache(approx_mode=False)
        time_provider_func = create_osrm_time_provider(cache)
        
        stops_by_depot = assign_stops_to_depots(
            stops=stops,
            depots=depots,
            time_provider=lambda lat1, lon1, lat2, lon2: time_provider_func(lat1, lon1, lat2, lon2)
        )
        
        for dc_id, assigned_stops in stops_by_depot.items():
            logger.info(f"  Depot {dc_id}: {len(assigned_stops)} stops")
        
        # Convert stops to dict format
        stops_by_depot_dict = {}
        stops_by_id = {}
        for dc_id, stop_list in stops_by_depot.items():
            stops_by_depot_dict[dc_id] = [
                {
                    "id": s.id,
                    "name": s.id,
                    "lat": s.lat,
                    "lon": s.lon,
                    "coordinates": [s.lat, s.lon],
                    "demand": s.demand,
                    "households": s.demand
                }
                for s in stop_list
            ]
            for s in stop_list:
                stops_by_id[s.id] = {
                    "id": s.id,
                    "lat": s.lat,
                    "lon": s.lon,
                    "demand": s.demand,
                    "households": s.demand,
                    "service_time_s": s.service_time
                }
        
        # Create OSRM providers
        time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
        
        # Generate baseline solution once (used by both experiments)
        print("\n" + "=" * 80)
        print("Generating baseline VRP solution...")
        print("=" * 80)
        baseline_solution = solve_multi_depot(
            depots=depots,
            vehicles_by_depot=vehicles,
            stops_by_depot=stops_by_depot_dict,
            request_geometry=False
        )
        baseline_solution["depots"] = depots
        baseline_solution["stops_dict"] = stops_by_id
        logger.info(f"Baseline solution generated with {len(baseline_solution.get('routes_by_dc', {}))} DCs")
    
    # Initialize output_dir (needed for both branches)
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    if not args.reuse:
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: Using Z3 = MAD (Notion spec)")
        print("=" * 80)
        results_MAD = run_experiment_with_Z3_mode(
            depots, vehicles, stops_by_depot_dict, stops_by_id,
            time_provider, distance_provider, config, use_MAD=True
        )
        
        # Save MAD debug info to separate file to avoid conflicts
        mad_debug_file = output_dir / "variance_vs_mad_MAD_debug.json"
        with open(mad_debug_file, 'w') as f:
            json.dump(results_MAD.get("debug_info", {}), f, indent=2)
        logger.info(f"Saved MAD debug info: {mad_debug_file}")
        
        print(f"\nBaseline (MAD) - RAW scores:")
        print(f"  Z1 (RAW): {results_MAD['baseline']['Z1']:.2f}")
        print(f"  Z2 (RAW): {results_MAD['baseline']['Z2']:.2f}")
        print(f"  Z3 (RAW): {results_MAD['baseline']['Z3']:.2f}")
        print(f"  Z (normalized): {results_MAD['baseline']['Z']:.4f}")
        
        print(f"\nImproved (MAD) - RAW scores:")
        print(f"  Z1 (RAW): {results_MAD['improved']['Z1']:.2f} ({results_MAD['improvement']['Z1_pct']:+.1f}%)")
        print(f"  Z2 (RAW): {results_MAD['improved']['Z2']:.2f} ({results_MAD['improvement']['Z2_pct']:+.1f}%)")
        print(f"  Z3 (RAW): {results_MAD['improved']['Z3']:.2f} ({results_MAD['improvement']['Z3_pct']:+.1f}%)")
        print(f"  Z (normalized): {results_MAD['improved']['Z']:.4f} ({results_MAD['improvement']['Z_pct']:+.1f}%)")
    
    # Run experiment with VARIANCE
    if not args.reuse:
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Using Z3 = Variance (current implementation)")
        print("=" * 80)
        results_variance = run_experiment_with_Z3_mode(
            depots, vehicles, stops_by_depot_dict, stops_by_id,
            time_provider, distance_provider, config, use_MAD=False
        )
    else:
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: Using Z3 = Variance (reuse baseline/MAD, train variance only)")
        print("=" * 80)
        # Already have depots/vehicles/stops_by_depot_dict/stops_by_id from reuse branch
        cache = iNaviCache(approx_mode=False)
        time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
        results_variance = run_experiment_with_Z3_mode(
            depots, vehicles, stops_by_depot_dict, stops_by_id,
            time_provider, distance_provider, config, use_MAD=False
        )
    
    # Save Variance debug info to separate file to avoid conflicts
    variance_debug_file = output_dir / "variance_vs_mad_VARIANCE_debug.json"
    with open(variance_debug_file, 'w') as f:
        json.dump(results_variance.get("debug_info", {}), f, indent=2)
    logger.info(f"Saved Variance debug info: {variance_debug_file}")
    
    print(f"\nBaseline (Variance) - RAW scores:")
    print(f"  Z1 (RAW): {results_variance['baseline']['Z1']:.2f}")
    print(f"  Z2 (RAW): {results_variance['baseline']['Z2']:.2f}")
    print(f"  Z3 (RAW): {results_variance['baseline']['Z3']:.2f}")
    print(f"  Z (normalized): {results_variance['baseline']['Z']:.4f}")
    
    print(f"\nImproved (Variance) - RAW scores:")
    print(f"  Z1 (RAW): {results_variance['improved']['Z1']:.2f} ({results_variance['improvement']['Z1_pct']:+.1f}%)")
    print(f"  Z2 (RAW): {results_variance['improved']['Z2']:.2f} ({results_variance['improvement']['Z2_pct']:+.1f}%)")
    print(f"  Z3 (RAW): {results_variance['improved']['Z3']:.2f} ({results_variance['improvement']['Z3_pct']:+.1f}%)")
    print(f"  Z (normalized): {results_variance['improved']['Z']:.4f} ({results_variance['improvement']['Z_pct']:+.1f}%)")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Variance vs MAD")
    print("=" * 80)
    
    print(f"\nBaseline Z3 values:")
    print(f"  Variance: {results_variance['baseline']['Z3']:.2f}")
    print(f"  MAD:      {results_MAD['baseline']['Z3']:.2f}")
    if results_MAD['baseline']['Z3'] > 0:
        print(f"  Ratio (Variance/MAD): {results_variance['baseline']['Z3'] / results_MAD['baseline']['Z3']:.2f}x")
    else:
        print(f"  Ratio (Variance/MAD): N/A (MAD is 0)")
    
    print(f"\nImproved Z3 values:")
    print(f"  Variance: {results_variance['improved']['Z3']:.2f}")
    print(f"  MAD:      {results_MAD['improved']['Z3']:.2f}")
    if results_MAD['improved']['Z3'] > 0:
        print(f"  Ratio (Variance/MAD): {results_variance['improved']['Z3'] / results_MAD['improved']['Z3']:.2f}x")
    else:
        print(f"  Ratio (Variance/MAD): N/A (MAD is 0)")
    
    print(f"\nZ3 Improvement:")
    print(f"  Variance: {results_variance['improvement']['Z3_pct']:+.1f}%")
    print(f"  MAD:      {results_MAD['improvement']['Z3_pct']:+.1f}%")
    print(f"  Difference: {abs(results_variance['improvement']['Z3_pct'] - results_MAD['improvement']['Z3_pct']):.1f}%")
    
    print(f"\nCombined Z Improvement:")
    print(f"  Variance: {results_variance['improvement']['Z_pct']:+.1f}%")
    print(f"  MAD:      {results_MAD['improvement']['Z_pct']:+.1f}%")
    print(f"  Difference: {abs(results_variance['improvement']['Z_pct'] - results_MAD['improvement']['Z_pct']):.1f}%")
    
    # Save results
    output_file = Path("outputs") / "variance_vs_mad_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {
        "config": config,
        "variance": {
            "baseline": results_variance["baseline"],
            "improved": results_variance["improved"],
            "improvement": results_variance["improvement"]
        },
        "MAD": {
            "baseline": results_MAD["baseline"],
            "improved": results_MAD["improved"],
            "improvement": results_MAD["improvement"]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)
    
    # Save scores to CSV (similar to compare_waiting_and_scores.py)
    import csv
    scores_file = output_dir / "compare_scores_z3.csv"
    with open(scores_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["method", "Z1", "Z2", "Z3", "Z"])
        # Baseline (same for both, use MAD baseline)
        w.writerow(["baseline", results_MAD["baseline"]["Z1"], results_MAD["baseline"]["Z2"], 
                   results_MAD["baseline"]["Z3"], results_MAD["baseline"]["Z"]])
        # Variance improved
        w.writerow(["Variance", results_variance["improved"]["Z1"], results_variance["improved"]["Z2"],
                   results_variance["improved"]["Z3"], results_variance["improved"]["Z"]])
        # MAD improved
        w.writerow(["MAD", results_MAD["improved"]["Z1"], results_MAD["improved"]["Z2"],
                   results_MAD["improved"]["Z3"], results_MAD["improved"]["Z"]])
    logger.info(f"Saved Z-scores: {scores_file}")
    
    # Generate waiting time histograms
    print("\n" + "=" * 80)
    print("Generating waiting time histograms...")
    print("=" * 80)
    
    # Extract waiting times (weighted by households/demand)
    # Use baseline_solution (loaded from file if reusing, or generated if not)
    baseline_waiting = compute_waiting_times(baseline_solution, stops_by_id, time_provider)
    mad_waiting = compute_waiting_times(results_MAD["solution"], stops_by_id, time_provider)
    variance_waiting = compute_waiting_times(results_variance["solution"], stops_by_id, time_provider)
    
    # Collect raw waits and their household weights (like compare_waiting_and_scores.py)
    # X-axis: raw waiting time (seconds)
    # Frequencies: weighted by households (or demand if households missing)
    def _raw_waits_with_weights(waiting):
        vals, wts = [], []
        for stop_id, w in waiting.items():
            wt = stops_by_id.get(stop_id, {}).get("households", stops_by_id.get(stop_id, {}).get("demand", 1))
            if wt in (None, 0):
                wt = 1
            vals.append(w)  # Raw waiting time
            wts.append(wt)  # Household weight
        return vals, wts

    baseline_waits, baseline_wts = _raw_waits_with_weights(baseline_waiting)
    variance_waits, variance_wts = _raw_waits_with_weights(variance_waiting)
    mad_waits, mad_wts = _raw_waits_with_weights(mad_waiting)
    
    # Save wait values to CSV (similar to compare_waiting_and_scores.py)
    waits_file = output_dir / "compare_wait_values_z3.csv"
    with open(waits_file, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["method", "waiting_time_seconds", "households"])
        # Get raw waiting times and household weights for each method
        for stop_id, w_raw in baseline_waiting.items():
            wt = stops_by_id.get(stop_id, {}).get("households", stops_by_id.get(stop_id, {}).get("demand", 1))
            w.writerow(["baseline", w_raw, wt])
        for stop_id, w_raw in variance_waiting.items():
            wt = stops_by_id.get(stop_id, {}).get("households", stops_by_id.get(stop_id, {}).get("demand", 1))
            w.writerow(["Variance", w_raw, wt])
        for stop_id, w_raw in mad_waiting.items():
            wt = stops_by_id.get(stop_id, {}).get("households", stops_by_id.get(stop_id, {}).get("demand", 1))
            w.writerow(["MAD", w_raw, wt])
    logger.info(f"Saved wait values for plotting: {waits_file}")
    
    # Plot histograms
    hist_output = Path("outputs") / "variance_vs_mad_wait_hist"
    plot_Z3_comparison_histograms(
        baseline_waits=baseline_waits,
        variance_waits=variance_waits,
        mad_waits=mad_waits,
        city_name="daejeon",  # Could be configurable
        output_path=hist_output,
        baseline_weights=baseline_wts,
        variance_weights=variance_wts,
        mad_weights=mad_wts,
    )
    
    print(f"Waiting time histograms saved to: {hist_output}.png (and .html if Plotly available)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    ✓ Both experiments completed successfully.
    
    Key Observations:
    1. Z3 (Variance) values are typically much larger than Z3 (MAD)
       - Variance penalizes outliers more heavily (quadratic vs linear)
    
    2. The optimization behavior may differ:
       - Variance: Focuses more on reducing extreme deviations
       - MAD: Focuses more on overall fairness distribution
    
    3. Combined Z improvement:
       - Variance: {results_variance['improvement']['Z_pct']:+.1f}%
       - MAD:      {results_MAD['improvement']['Z_pct']:+.1f}%
    
    Recommendation:
    - Use MAD (compute_Z3_MAD) to match Notion spec exactly
    - Variance is acceptable as approximation but not spec-compliant
    """)


if __name__ == "__main__":
    main()

