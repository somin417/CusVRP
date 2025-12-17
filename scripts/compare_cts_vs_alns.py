#!/usr/bin/env python3
"""
Compare ALNS with fixed operators vs ALNS with Contextual Thompson Sampling (CTS).

This script follows the same pipeline as compare_waiting_and_scores.py:
- Reuses ALNS (fixed) results from compare_waiting_and_scores.py if available
- Runs CTS with same settings (depots, stops, parameters)
- Compares Z scores and waiting time distributions
- Uses MAD for Z3 (spec-compliant)

Outputs:
  - cts_vs_alns_scores.csv: Z1, Z2, Z3 (MAD), Z for baseline, ALNS, CTS
  - cts_vs_alns_wait_panels.png: four-panel waiting-time comparison
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from src.vrp_fairness.objectives import (
    compute_waiting_times,
    compute_Z1,
    compute_Z2,
    compute_Z3_MAD,
    compute_combined_Z,
)
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache
from scripts.utils.plotting_utils import calc_bins, smooth_curve

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def calc_scores(
    solution: Dict[str, Any],
    depots: List[Dict[str, Any]],
    use_distance: bool,
    alpha: float,
    beta: float,
    gamma: float,
    Z1_star: Optional[float] = None,
    Z2_star: Optional[float] = None,
    Z3_star: Optional[float] = None,
    time_provider: Optional[Any] = None,
    distance_provider: Optional[Any] = None,
    cache: Optional[Any] = None,
    baseline_stops_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Calculate Z-scores for a solution using MAD for Z3 (spec).
    
    Args:
        solution: Solution dictionary
        depots: List of depot dictionaries
        use_distance: Whether to use distance for Z2
        alpha, beta, gamma: Weights for combined Z
        Z1_star, Z2_star, Z3_star: Normalizers (if None, will use solution's own values)
        time_provider: Optional time provider (if None, will create new one)
        distance_provider: Optional distance provider (if None, will create new one)
        cache: Optional cache (if None, will create new one)
        baseline_stops_dict: Baseline stops_dict to use for consistency
    
    Returns:
        Dictionary with waiting times, Z1, Z2, Z3, Z, and Z_star
    """
    if cache is None:
        cache = iNaviCache(approx_mode=False)
    
    # Always use baseline stops_dict for consistency
    if baseline_stops_dict is None:
        baseline_stops_dict = solution.get("stops_dict", {})
    stops_by_id = baseline_stops_dict
    
    # Use provided providers or create new ones
    if time_provider is None or distance_provider is None:
        from src.vrp_fairness.osrm_provider import create_osrm_providers
        tp, dp = create_osrm_providers(depots, stops_by_id, cache)
        if time_provider is None:
            time_provider = tp
        if distance_provider is None:
            distance_provider = dp
    
    waiting = compute_waiting_times(solution, stops_by_id, time_provider)
    Z1 = compute_Z1(waiting, stops_by_id)
    Z2 = compute_Z2(solution, distance_provider, time_provider, use_distance)
    Z3 = compute_Z3_MAD(waiting, stops_by_id)  # Always use MAD
    
    # If normalizers not provided, use solution's own values (for baseline)
    if Z1_star is None:
        Z1_star, Z2_star, Z3_star = Z1, Z2, Z3
    
    Z = compute_combined_Z(Z1, Z2, Z3, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    return {
        "waiting": waiting,
        "Z1": Z1,
        "Z2": Z2,
        "Z3": Z3,
        "Z": Z,
        "Z_star": (Z1_star, Z2_star, Z3_star),
    }


# Use common utilities from plotting_utils (no aliases needed)


def plot_wait_panels(out_path: Path, city: str, baseline_waits, alns_waits, cts_waits):
    """
    Four panels with consistent binning and bar widths:
      - Baseline only
      - ALNS (fixed) only
      - CTS only
      - Combined comparison
    Each panel overlays smooth curves through bar centers.
    """
    bins = calc_bins(baseline_waits, alns_waits, cts_waits)
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
    centers = bins[:-1] + bin_width / 2
    bar_width = bin_width * 0.8

    colors = {
        "Baseline": "#4C78A8",
        "ALNS (fixed)": "#54A24B",
        "CTS": "#F58518",
    }

    def _counts(data, weights=None):
        return np.histogram(data, bins=bins, weights=weights)[0]

    baseline_counts = _counts(baseline_waits, baseline_weights)
    alns_counts = _counts(alns_waits, alns_weights)
    cts_counts = _counts(cts_waits, cts_weights)

    # Use a single padded cap across all panels for consistent y-scale (like compare_waiting_and_scores.py)
    global_max = 0
    for arr in (baseline_counts, alns_counts, cts_counts):
        if len(arr) > 0:
            global_max = max(global_max, max(arr))
    global_cap = global_max * 1.05 if global_max > 0 else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("Baseline", baseline_waits, baseline_counts, axes[0, 0]),
        ("ALNS (fixed)", alns_waits, alns_counts, axes[0, 1]),
        ("CTS", cts_waits, cts_counts, axes[1, 0]),
    ]

    for title, waits, counts, ax in panels:
        # Apply shared padded cap so all three single panels align
        panel_cap = global_cap
        ax.bar(centers, counts, width=bar_width, color=colors[title], alpha=0.75, edgecolor="white", linewidth=0.7)
        # Add smooth curve only
        x_smooth, y_smooth = smooth_curve(centers, counts)
        if panel_cap is not None:
            y_smooth = np.minimum(y_smooth, panel_cap)
        ax.plot(x_smooth, y_smooth, color=colors[title], linewidth=2.0, linestyle="-", alpha=0.9)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Waiting time (seconds)")
        ax.set_ylabel("Frequency (household-weighted)")
        if panel_cap is not None:
            ax.set_ylim(top=panel_cap)
        ax.grid(alpha=0.3, axis="y")

    # Combined panel
    axc = axes[1, 1]
    offset = bin_width * 0.25
    series = [
        ("Baseline", baseline_counts, -offset),
        ("ALNS (fixed)", alns_counts, 0.0),
        ("CTS", cts_counts, offset),
    ]
    for name, counts, offs in series:
        axc.bar(centers + offs, counts, width=bar_width / 3, color=colors[name], alpha=0.75, edgecolor="white", linewidth=0.7, label=name)
        # Add smooth curve only
        x_smooth, y_smooth = smooth_curve(centers, counts)
        # Use shared padded cap for combined panel as well
        local_cap = global_cap
        if local_cap is not None:
            y_smooth = np.minimum(y_smooth, local_cap)
        axc.plot(x_smooth + offs, y_smooth, color=colors[name], linewidth=2.0, linestyle="-", alpha=0.9)
    axc.set_title("All Combined", fontweight="bold")
    axc.set_xlabel("Waiting time (seconds)")
    axc.set_ylabel("Frequency (household-weighted)")
    if global_cap is not None:
        axc.set_ylim(top=global_cap)
    axc.legend(loc="upper right")
    axc.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Waiting Time Distribution (household-weighted frequency) — {city.title()}", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved waiting panels: {out_path}")
    return bins, baseline_counts, alns_counts, cts_counts


def save_wait_hist_interactive(out_path: Path, city: str, baseline_waits, alns_waits=None, cts_waits=None):
    """Deprecated: HTML generation removed. Only PNG plots are generated now."""
    # HTML generation removed - only PNG plots are generated
    # Maps are the only place where HTML is generated
    pass


def main():
    p = argparse.ArgumentParser(description="Compare ALNS (fixed) vs CTS")
    p.add_argument("--gpkg", default="data/yuseong_housing_3__point.gpkg")
    p.add_argument("--layer", default="yuseong_housing_2__point")
    p.add_argument("--sample-n", type=int, default=50)
    p.add_argument("--num-dcs", type=int, default=3)
    p.add_argument("--city", default="daejeon")
    p.add_argument("--demand-field", default="A26")
    p.add_argument("--eps", type=float, default=0.30, help="Cost budget tolerance (default: 0.30 = 130% of baseline Z2). Note: Higher than other scripts (0.10) to allow more exploration for CTS comparison.")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-distance-objective", action="store_true")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--output-dir", default="outputs", help="Output directory (should match compare_waiting_and_scores.py to reuse ALNS results)")
    p.add_argument("--force-rerun-alns", action="store_true", help="Force rerun ALNS even if results exist")
    p.add_argument("--cts-only", action="store_true", help="Run CTS only (skip ALNS, requires baseline.json)")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)
    
    # Create organized subdirectories
    solutions_dir = out / "solutions"
    data_dir = out / "data"
    plots_dir = out / "plots"
    debug_dir = out / "debug"
    solutions_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    debug_dir.mkdir(exist_ok=True)

    # Try to reuse ALNS results from compare_waiting_and_scores.py
    # Files created by compare_waiting_and_scores.py:
    # - baseline.json: baseline solution (created by run_experiment --method baseline)
    # - ALNS_MAD.json: ALNS solution (created by run_experiment --method proposed --operator-mode fixed)
    # - alns_mad_debug.json: ALNS debug info (created by run_experiment when method=proposed)
    baseline = None
    alns_solution = None
    alns_debug = None
    depots = None
    
    if not args.force_rerun_alns:
        # Try to load baseline and ALNS results from compare_waiting_and_scores.py
        # Files created by compare_waiting_and_scores.py:
        # - baseline.json: baseline solution
        # - ALNS_MAD.json: ALNS solution
        # - alns_mad_debug.json: ALNS debug info (ONLY created when method=proposed)
        # - solutions/{run_id}_proposed.json: ALNS solution backup (ONLY created when method=proposed)
        
        baseline_file = solutions_dir / "baseline.json"
        alns_debug_file = debug_dir / "alns_mad_debug.json"  # This file ONLY exists if ALNS was run
        
        # Initialize alns_solution_file (default to ALNS_MAD.json)
        alns_solution_file = solutions_dir / "ALNS_MAD.json"
        
                    # Verify that alns_mad_debug.json exists (guarantees ALNS was run)
        if baseline_file.exists() and alns_debug_file.exists() and alns_solution_file.exists():
            try:
                baseline = load_json(baseline_file)
                alns_debug = load_json(alns_debug_file)
                depots = baseline.get("depots", [])
                
                # Verify it's actually ALNS by checking debug info has best_objectives
                if alns_debug.get("best_objectives") and alns_debug.get("normalizers"):
                    # Load ALNS solution
                    alns_solution = load_json(alns_solution_file)
                    
                    # Additional verification: check file modification times
                    # alns_mad_debug.json should be newer than or equal to ALNS_MAD.json
                    # (if using ALNS_MAD.json)
                    if alns_solution_file == solutions_dir / "ALNS_MAD.json":
                        debug_mtime = alns_debug_file.stat().st_mtime
                        solution_mtime = alns_solution_file.stat().st_mtime
                        if solution_mtime < debug_mtime - 1:  # Allow 1 second tolerance
                            logger.warning(f"ALNS_MAD.json ({solution_mtime}) is older than alns_mad_debug.json ({debug_mtime})")
                            logger.warning("This suggests ALNS_MAD.json might contain local search results, not ALNS")
                            logger.info("Will run ALNS from scratch to be safe")
                            baseline = None
                            alns_solution = None
                            alns_debug = None
                        else:
                            logger.info("Reusing ALNS results from compare_waiting_and_scores.py")
                            logger.info(f"  Baseline: {baseline_file}")
                            logger.info(f"  ALNS solution: {alns_solution_file}")
                            logger.info(f"  ALNS debug: {alns_debug_file}")
                    else:
                        # Using solutions/ file, which is guaranteed to be ALNS
                        logger.info("Reusing ALNS results from compare_waiting_and_scores.py")
                        logger.info(f"  Baseline: {baseline_file}")
                        logger.info(f"  ALNS solution: {alns_solution_file}")
                        logger.info(f"  ALNS debug: {alns_debug_file}")
                else:
                    logger.warning("alns_mad_debug.json exists but doesn't contain expected ALNS data")
                    logger.info("Will run ALNS from scratch")
                    baseline = None
                    alns_solution = None
                    alns_debug = None
            except Exception as e:
                logger.warning(f"Failed to load existing results: {e}")
                logger.info("Will run ALNS from scratch")
                baseline = None
                alns_solution = None
                alns_debug = None
        else:
            missing = []
            if not baseline_file.exists():
                missing.append("baseline.json")
            if not alns_debug_file.exists():
                missing.append("alns_mad_debug.json")
            if not alns_solution_file.exists():
                missing.append(f"{alns_solution_file.name}")
            logger.info(f"Missing files for ALNS reuse: {', '.join(missing)}")
            logger.info("Will run ALNS from scratch")
    
    # If --cts-only, skip ALNS and use baseline.json only
    if args.cts_only:
        if baseline is None:
            baseline_file = solutions_dir / "baseline.json"
            if not baseline_file.exists():
                logger.error("--cts-only requires baseline.json to exist")
                sys.exit(1)
            baseline = load_json(baseline_file)
            depots = baseline.get("depots", [])
            if not depots:
                raise RuntimeError("No depots found in baseline.json")
            logger.info("Using baseline.json for CTS-only run")
        # Skip ALNS, go directly to CTS
        alns_solution = None
        alns_debug = None
    
    # If we don't have ALNS results and not --cts-only, run ALNS first (same as compare_waiting_and_scores.py)
    elif baseline is None or alns_solution is None or args.force_rerun_alns:
        logger.info("Running ALNS (fixed) from scratch...")
        
        # Common base command parts
        base_cmd = [
            "python",
            "-m",
            "src.vrp_fairness.run_experiment",
            "--seed",
            str(args.seed),
            "--gpkg",
            args.gpkg,
            "--layer",
            args.layer,
            "--sample-n",
            str(args.sample_n),
            "--city",
            args.city,
            "--demand-field",
            args.demand_field,
            "--eps",
            str(args.eps),
            "--iters",
            str(args.iters),
            "--alpha",
            str(args.alpha),
            "--beta",
            str(args.beta),
            "--gamma",
            str(args.gamma),
            "--output-dir",
            str(args.output_dir),
            "--no-plots",
        ]
        
        if args.use_distance_objective:
            base_cmd.append("--use-distance-objective")
        
        # Run baseline first
        cmd_base = base_cmd.copy()
        iters_idx = cmd_base.index("--iters")
        cmd_base[iters_idx + 1] = "1"
        output_dir_idx = cmd_base.index("--output-dir")
        predefined_dcs = [
            {"id": "DC_Logen", "lat": 36.3800587, "lon": 127.3777765},
            {"id": "DC_Hanjin", "lat": 36.3711833, "lon": 127.4050933},
            {"id": "DC_CJ", "lat": 36.449416, "lon": 127.4070349},
        ]
        dcs_args = []
        if args.num_dcs == 3:
            for depot in predefined_dcs:
                dcs_args.append(f"{depot['lat']},{depot['lon']},{depot['id']}")
        if dcs_args:
            cmd_base.insert(output_dir_idx, "--dcs")
            for dc_str in reversed(dcs_args):
                cmd_base.insert(output_dir_idx + 1, dc_str)
        else:
            cmd_base.insert(output_dir_idx, "--num-dcs")
            cmd_base.insert(output_dir_idx + 1, str(args.num_dcs))
        cmd_base += ["--method", "baseline"]
        logger.info("Running baseline...")
        subprocess.check_call(cmd_base)
        
        # Load baseline to extract DCs
        baseline = load_json(solutions_dir / "baseline.json")
        depots = baseline.get("depots", [])
        if not depots:
            raise RuntimeError("No depots found in baseline.json")
        
        # Convert depots to --dcs format
        dcs_args = []
        for depot in depots:
            dc_str = f"{depot['lat']},{depot['lon']}"
            if depot.get("name") or depot.get("id"):
                dc_str += f",{depot.get('name') or depot.get('id')}"
            dcs_args.append(dc_str)
        
        # Run ALNS (fixed)
        cmd_alns = base_cmd.copy()
        output_dir_idx = cmd_alns.index("--output-dir")
        cmd_alns.insert(output_dir_idx, "--dcs")
        for dc_str in reversed(dcs_args):
            cmd_alns.insert(output_dir_idx + 1, dc_str)
        cmd_alns += ["--method", "proposed", "--operator-mode", "fixed", "--use-mad"]
        logger.info("Running ALNS (fixed)...")
        subprocess.check_call(cmd_alns)
        
        # Load ALNS results
        alns_solution = load_json(solutions_dir / "ALNS_MAD.json")
        alns_debug_file = debug_dir / "alns_mad_debug.json"
        if alns_debug_file.exists():
            alns_debug = load_json(alns_debug_file)
    
    # Now run CTS with same settings
    logger.info("Running CTS...")
    cmd_cts = [
        "python",
        "-m",
        "src.vrp_fairness.run_experiment",
        "--seed",
        str(args.seed),
        "--gpkg",
        args.gpkg,
        "--layer",
        args.layer,
        "--sample-n",
        str(args.sample_n),
        "--city",
        args.city,
        "--demand-field",
        args.demand_field,
        "--eps",
        str(args.eps),
        "--iters",
        str(args.iters),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--gamma",
        str(args.gamma),
        "--output-dir",
        str(args.output_dir),  # Use main output dir, CTS will save to cts_solution.json
        "--no-plots",
        "--method", "proposed",
        "--operator-mode", "cts",
        "--use-mad",
    ]
    
    # Add DCs
    dcs_args = []
    for depot in depots:
        dc_str = f"{depot['lat']},{depot['lon']}"
        if depot.get("name") or depot.get("id"):
            dc_str += f",{depot.get('name') or depot.get('id')}"
        dcs_args.append(dc_str)
    
    output_dir_idx = cmd_cts.index("--output-dir")
    cmd_cts.insert(output_dir_idx, "--dcs")
    for dc_str in reversed(dcs_args):
        cmd_cts.insert(output_dir_idx + 1, dc_str)
    
    if args.use_distance_objective:
        cmd_cts.append("--use-distance-objective")
    
    subprocess.check_call(cmd_cts)
    
    # Load CTS results from main output directory
    # CTS saves to cts_solution.json and cts_debug.json (separate from ALNS files)
    cts_solution = load_json(solutions_dir / "cts_solution.json")
    cts_debug_file = debug_dir / "cts_debug.json"  # CTS now saves directly to cts_debug.json
    cts_debug = None
    if cts_debug_file.exists():
        cts_debug = load_json(cts_debug_file)
        logger.info(f"Loaded CTS debug info from: {cts_debug_file}")
    
    # Ensure CTS solution has stops_dict and depots for consistency
    if "stops_dict" not in cts_solution:
        cts_solution["stops_dict"] = baseline.get("stops_dict", {})
    if "depots" not in cts_solution:
        cts_solution["depots"] = depots
    
    # CRITICAL: Verify that CTS used the same stops/demands as ALNS
    # CTS uses the same baseline.json, so we can verify using baseline
    baseline_stops_dict = baseline.get("stops_dict", {})
    cts_baseline_stops_dict = baseline.get("stops_dict", {})  # CTS uses same baseline
    
    if baseline_stops_dict and cts_baseline_stops_dict:
        # Compare stop IDs and demands
        baseline_stop_ids = set(baseline_stops_dict.keys())
        cts_stop_ids = set(cts_baseline_stops_dict.keys())
        
        if baseline_stop_ids != cts_stop_ids:
            missing_in_cts = baseline_stop_ids - cts_stop_ids
            extra_in_cts = cts_stop_ids - baseline_stop_ids
            logger.error("="*70)
            logger.error("STOP MISMATCH DETECTED!")
            logger.error("="*70)
            logger.error(f"CTS run used different stops than ALNS run!")
            logger.error(f"  Missing in CTS: {len(missing_in_cts)} stops")
            logger.error(f"  Extra in CTS: {len(extra_in_cts)} stops")
            if missing_in_cts:
                logger.error(f"  Missing stop IDs: {list(missing_in_cts)[:10]}...")
            if extra_in_cts:
                logger.error(f"  Extra stop IDs: {list(extra_in_cts)[:10]}...")
            logger.error("="*70)
            logger.error("This comparison is INVALID! Stops must match.")
            logger.error("Ensure --seed, --sample-n, --gpkg, --layer match compare_waiting_and_scores.py")
            raise RuntimeError("Stop mismatch: CTS and ALNS used different stops")
        
        # Compare demands for each stop
        demand_mismatches = []
        for stop_id in baseline_stop_ids:
            baseline_demand = baseline_stops_dict.get(stop_id, {}).get("demand", 0)
            cts_demand = cts_baseline_stops_dict.get(stop_id, {}).get("demand", 0)
            if abs(baseline_demand - cts_demand) > 0.001:
                demand_mismatches.append((stop_id, baseline_demand, cts_demand))
        
        if demand_mismatches:
            logger.warning(f"Demand mismatches found for {len(demand_mismatches)} stops:")
            for stop_id, base_d, cts_d in demand_mismatches[:5]:
                logger.warning(f"  {stop_id}: baseline={base_d}, cts={cts_d}")
            if len(demand_mismatches) > 5:
                logger.warning(f"  ... and {len(demand_mismatches) - 5} more")
        else:
            logger.info("✓ Verified: CTS used same stops and demands as ALNS")
    else:
        logger.warning("Could not verify stops match (stops_dict missing in baseline or CTS baseline)")
    
    # Create shared OSRM providers for consistent calculation
    # Always use ALNS baseline stops_dict for all calculations
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, baseline_stops_dict, cache)
    
    # Compute scores using same providers and stops_dict for consistency
    # Baseline: use its own values as normalizers
    base_scores = calc_scores(
        baseline, depots, args.use_distance_objective, args.alpha, args.beta, args.gamma,
        time_provider=time_provider, distance_provider=distance_provider, cache=cache,
        baseline_stops_dict=baseline_stops_dict
    )
    
    # Use normalizers from ALNS debug if available (for consistency)
    if alns_debug and "normalizers" in alns_debug:
        normalizers = alns_debug["normalizers"]
        Z1_star = normalizers.get("Z1_star", base_scores["Z1"])
        Z2_star = normalizers.get("Z2_star", base_scores["Z2"])
        Z3_star = normalizers.get("Z3_star", base_scores["Z3"])
        logger.info(f"Using normalizers from ALNS: Z1*={Z1_star:.1f}, Z2*={Z2_star:.1f}, Z3*={Z3_star:.1f}")
        
        # Use baseline_objectives from ALNS debug for baseline scores
        baseline_objectives = alns_debug.get("baseline_objectives", {})
        if baseline_objectives and all(k in baseline_objectives for k in ["Z1", "Z2", "Z3"]):
            Z1_baseline = baseline_objectives.get("Z1")
            Z2_baseline = baseline_objectives.get("Z2")
            Z3_baseline = baseline_objectives.get("Z3")
            logger.info(f"Using baseline objectives from ALNS: Z1={Z1_baseline:.1f}, Z2={Z2_baseline:.1f}, Z3={Z3_baseline:.1f}")
            
            from src.vrp_fairness.objectives import compute_combined_Z
            Z_baseline = compute_combined_Z(Z1_baseline, Z2_baseline, Z3_baseline,
                                          Z1_star, Z2_star, Z3_star,
                                          args.alpha, args.beta, args.gamma)
            base_scores = {
                "Z1": Z1_baseline,
                "Z2": Z2_baseline,
                "Z3": Z3_baseline,
                "Z": Z_baseline,
                "waiting": base_scores["waiting"],
            }
    else:
        Z1_star, Z2_star, Z3_star = base_scores["Z_star"]
        logger.info(f"Using normalizers from baseline: Z1*={Z1_star:.1f}, Z2*={Z2_star:.1f}, Z3*={Z3_star:.1f}")
    
    # ALNS: use best_objectives from debug if available
    alns_scores = None
    if alns_debug:
        best_objectives = alns_debug.get("best_objectives", {})
        if best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]):
            Z1_alns = best_objectives.get("Z1")
            Z2_alns = best_objectives.get("Z2")
            Z3_alns = best_objectives.get("Z3")
            
            if Z1_alns is not None and Z2_alns is not None and Z3_alns is not None:
                logger.info(f"Using RAW best objectives from ALNS:")
                logger.info(f"  Z1_alns (RAW)={Z1_alns:.2f}")
                logger.info(f"  Z2_alns (RAW)={Z2_alns:.2f}")
                logger.info(f"  Z3_alns (RAW)={Z3_alns:.2f}")
                
                from src.vrp_fairness.objectives import compute_combined_Z
                Z_alns = compute_combined_Z(Z1_alns, Z2_alns, Z3_alns,
                                           Z1_star, Z2_star, Z3_star,
                                           args.alpha, args.beta, args.gamma)
                
                # Still need waiting times for plotting
                alns_scores_temp = calc_scores(
                    alns_solution, depots, args.use_distance_objective, args.alpha, args.beta, args.gamma,
                    Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
                    time_provider=time_provider, distance_provider=distance_provider, cache=cache,
                    baseline_stops_dict=baseline_stops_dict
                )
                
                alns_scores = {
                    "Z1": Z1_alns,
                    "Z2": Z2_alns,
                    "Z3": Z3_alns,
                    "Z": Z_alns,
                    "waiting": alns_scores_temp["waiting"],
                }
    
    # Fallback: recompute ALNS scores
    if alns_scores is None:
        logger.warning("best_objectives not found in ALNS debug, recomputing ALNS scores...")
        alns_scores = calc_scores(
            alns_solution, depots, args.use_distance_objective, args.alpha, args.beta, args.gamma,
            Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
            time_provider=time_provider, distance_provider=distance_provider, cache=cache,
            baseline_stops_dict=baseline_stops_dict
        )
    
    # CTS: use best_objectives from debug if available
    cts_scores = None
    if cts_debug:
        best_objectives = cts_debug.get("best_objectives", {})
        if best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]):
            Z1_cts = best_objectives.get("Z1")
            Z2_cts = best_objectives.get("Z2")
            Z3_cts = best_objectives.get("Z3")
            
            if Z1_cts is not None and Z2_cts is not None and Z3_cts is not None:
                logger.info(f"Using RAW best objectives from CTS:")
                logger.info(f"  Z1_cts (RAW)={Z1_cts:.2f}")
                logger.info(f"  Z2_cts (RAW)={Z2_cts:.2f}")
                logger.info(f"  Z3_cts (RAW)={Z3_cts:.2f}")
                
                from src.vrp_fairness.objectives import compute_combined_Z
                Z_cts = compute_combined_Z(Z1_cts, Z2_cts, Z3_cts,
                                         Z1_star, Z2_star, Z3_star,
                                         args.alpha, args.beta, args.gamma)
                
                # Still need waiting times for plotting
                cts_scores_temp = calc_scores(
                    cts_solution, depots, args.use_distance_objective, args.alpha, args.beta, args.gamma,
                    Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
                    time_provider=time_provider, distance_provider=distance_provider, cache=cache,
                    baseline_stops_dict=baseline_stops_dict
                )
                
                cts_scores = {
                    "Z1": Z1_cts,
                    "Z2": Z2_cts,
                    "Z3": Z3_cts,
                    "Z": Z_cts,
                    "waiting": cts_scores_temp["waiting"],
                }
    
    # Fallback: recompute CTS scores
    if cts_scores is None:
        logger.warning("best_objectives not found in CTS debug, recomputing CTS scores...")
        cts_scores = calc_scores(
            cts_solution, depots, args.use_distance_objective, args.alpha, args.beta, args.gamma,
            Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
            time_provider=time_provider, distance_provider=distance_provider, cache=cache,
            baseline_stops_dict=baseline_stops_dict
        )
    
    # Z-scores comparison
    rows = [
        ("baseline", base_scores["Z1"], base_scores["Z2"], base_scores["Z3"], base_scores["Z"]),
        ("ALNS (fixed)", alns_scores["Z1"], alns_scores["Z2"], alns_scores["Z3"], alns_scores["Z"]),
        ("CTS", cts_scores["Z1"], cts_scores["Z2"], cts_scores["Z3"], cts_scores["Z"]),
    ]

    with open(data_dir / "cts_vs_alns_scores.csv", "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["method", "Z1", "Z2", "Z3_MAD", "Z"])
        w.writerows(rows)
    logger.info(f"Saved Z-scores: {data_dir / 'cts_vs_alns_scores.csv'}")
    
    # Plot waits (household-weighted, like compare_waiting_and_scores.py)
    # X-axis: raw waiting time, Y-axis: household-weighted frequency
    def _raw_waits_with_weights(wait_dict):
        vals, wts = [], []
        for stop_id, w in wait_dict.items():
            stop_meta = baseline_stops_dict.get(stop_id, {})
            wt = stop_meta.get("households", stop_meta.get("demand", 1))
            if wt in (None, 0):
                wt = 1
            vals.append(w)  # Raw waiting time
            wts.append(wt)  # Household weight
        return vals, wts
    
    baseline_waits, baseline_wts = _raw_waits_with_weights(base_scores["waiting"])
    alns_waits, alns_wts = _raw_waits_with_weights(alns_scores["waiting"])
    cts_waits, cts_wts = _raw_waits_with_weights(cts_scores["waiting"])
    
    bins, base_counts, alns_counts, cts_counts = plot_wait_panels(
        out_path=plots_dir / "cts_vs_alns_wait_panels.png",
        city=args.city,
        baseline_waits=baseline_waits,
        alns_waits=alns_waits,
        cts_waits=cts_waits,
        baseline_weights=baseline_wts,
        alns_weights=alns_wts,
        cts_weights=cts_wts,
    )
        # HTML generation removed - only PNG plots are generated
        # Maps are the only place where HTML is generated
    
    # Persist plot-ready data (like compare_waiting_and_scores.py)
    waits_path = data_dir / "cts_vs_alns_wait_values.csv"
    with open(waits_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["method", "waiting_time_seconds", "households"])
        for stop_id, w_raw in base_scores["waiting"].items():
            wt = baseline_stops_dict.get(stop_id, {}).get("households", baseline_stops_dict.get(stop_id, {}).get("demand", 1))
            w.writerow(["baseline", w_raw, wt])
        for stop_id, w_raw in alns_scores["waiting"].items():
            wt = baseline_stops_dict.get(stop_id, {}).get("households", baseline_stops_dict.get(stop_id, {}).get("demand", 1))
            w.writerow(["ALNS (fixed)", w_raw, wt])
        for stop_id, w_raw in cts_scores["waiting"].items():
            wt = baseline_stops_dict.get(stop_id, {}).get("households", baseline_stops_dict.get(stop_id, {}).get("demand", 1))
            w.writerow(["CTS", w_raw, wt])
    logger.info(f"Saved wait values for plotting: {waits_path}")

    hist_path = data_dir / "cts_vs_alns_wait_hist_data.csv"
    with open(hist_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "bin_center", "baseline_count", "alns_count", "cts_count"])
        for i in range(len(bins) - 1):
            left = bins[i]
            right = bins[i + 1]
            center = (left + right) / 2
            w.writerow([left, right, center, base_counts[i], alns_counts[i], cts_counts[i]])
    logger.info(f"Saved histogram data for plotting: {hist_path}")
    
    logger.info("\n" + "="*70)
    logger.info("COMPARISON COMPLETE")
    logger.info("="*70)
    logger.info(f"Baseline Z: {base_scores['Z']:.4f}")
    logger.info(f"ALNS (fixed) Z: {alns_scores['Z']:.4f} ({((base_scores['Z'] - alns_scores['Z']) / base_scores['Z'] * 100):+.2f}%)")
    logger.info(f"CTS Z: {cts_scores['Z']:.4f} ({((base_scores['Z'] - cts_scores['Z']) / base_scores['Z'] * 100):+.2f}%)")
    logger.info(f"CTS vs ALNS: {((alns_scores['Z'] - cts_scores['Z']) / alns_scores['Z'] * 100):+.2f}% improvement")


if __name__ == "__main__":
    main()
