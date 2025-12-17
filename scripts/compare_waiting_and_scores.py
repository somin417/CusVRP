#!/usr/bin/env python3
"""
Compare waiting-time distributions and Z-scores across baseline, local, and
proposed (ALSM/ALNS) using the same scoring logic as
compare_Z3_variance_vs_MAD.py (Z3 = MAD).

Outputs:
  - baseline_local_alns_mad_scores.csv: Z1, Z2, Z3 (MAD), Z per method
  - baseline_local_alns_mad_metrics.csv: key metrics deltas vs baseline
  - compare_wait_panels.png: four-panel waiting-time comparison (bars + lines)
  - baseline_local_alns_mad_wait_values.csv: waiting time values for plotting
  - baseline_local_alns_mad_wait_hist_data.csv: histogram data for plotting

Notes:
  - Runs three experiments sequentially (baseline, local, proposed) using
    run_experiment entrypoint.
  - Reuses OSRM providers and baseline stops for consistent scoring.
  - Z3 always uses MAD (spec-compliant).
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
    with open(path, "r") as f:
        return json.load(f)


def calc_scores(
    solution: Dict[str, Any],
    depots: List[Dict[str, Any]],
    use_distance: bool,
    use_mad: bool,
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
        use_mad: Whether to use MAD for Z3
        alpha, beta, gamma: Weights for combined Z
        Z1_star, Z2_star, Z3_star: Normalizers (if None, will use solution's own values)
        time_provider: Optional time provider (if None, will create new one)
        distance_provider: Optional distance provider (if None, will create new one)
        cache: Optional cache (if None, will create new one)
        baseline_stops_dict: Baseline stops_dict to use for consistency (contains metadata like households)
    
    Returns:
        Dictionary with waiting times, Z1, Z2, Z3, Z, and Z_star
    """
    if cache is None:
        cache = iNaviCache(approx_mode=False)
    
    # Always use baseline stops_dict for consistency (contains metadata like households that shouldn't change)
    if baseline_stops_dict is None:
        baseline_stops_dict = solution.get("stops_dict", {})
    stops_by_id = baseline_stops_dict
    
    # Use provided providers or create new ones
    if time_provider is None or distance_provider is None:
        tp, dp = create_osrm_providers(depots, stops_by_id, cache)
        if time_provider is None:
            time_provider = tp
        if distance_provider is None:
            distance_provider = dp
    
    waiting = compute_waiting_times(solution, stops_by_id, time_provider)
    Z1 = compute_Z1(waiting, stops_by_id)
    Z2 = compute_Z2(solution, distance_provider, time_provider, use_distance)
    Z3 = compute_Z3_MAD(waiting, stops_by_id)
    
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


# Use common utilities from plotting_utils
_calc_bins = calc_bins
_smooth_curve = smooth_curve


def plot_wait_panels(
    out_path: Path,
    city: str,
    baseline_waits,
    local_waits,
    proposed_waits,
    baseline_weights=None,
    local_weights=None,
    proposed_weights=None,
):
    """
    Four panels with consistent binning and bar widths:
      - Baseline only
      - Local only
      - ALSM (MAD) only
      - Combined comparison
    Each panel overlays a line through bar centers for quick distribution reading.
    Frequencies are weighted by households (weights), x-axis is raw waiting time.
    """
    bins = calc_bins(baseline_waits, local_waits, proposed_waits)
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
    centers = bins[:-1] + bin_width / 2
    bar_width = bin_width * 0.8

    colors = {
        "Baseline": "#4C78A8",
        "Local": "#54A24B",
        "ALSM (MAD)": "#F58518",
    }

    def _counts(data, weights=None):
        return np.histogram(data, bins=bins, weights=weights)[0]

    baseline_counts = _counts(baseline_waits, baseline_weights)
    local_counts = _counts(local_waits, local_weights)
    proposed_counts = _counts(proposed_waits, proposed_weights)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panels = [
        ("Baseline", baseline_waits, baseline_counts, axes[0, 0]),
        ("Local", local_waits, local_counts, axes[0, 1]),
        ("ALSM (MAD)", proposed_waits, proposed_counts, axes[1, 0]),
    ]

    # Use a single padded cap across all panels for consistent y-scale
    global_max = 0
    for arr in (baseline_counts, local_counts, proposed_counts):
        if len(arr) > 0:
            global_max = max(global_max, max(arr))
    global_cap = global_max * 1.05 if global_max > 0 else None

    for title, waits, counts, ax in panels:
        # Apply shared padded cap so all three single panels align
        panel_cap = global_cap
        ax.bar(centers, counts, width=bar_width, color=colors[title], alpha=0.75, edgecolor="white", linewidth=0.7)
        # Add smooth curve only (no original points)
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
        ("Local", local_counts, 0.0),
        ("ALSM (MAD)", proposed_counts, offset),
    ]
    max_combined = 0
    for name, counts, offs in series:
        axc.bar(centers + offs, counts, width=bar_width / 3, color=colors[name], alpha=0.75, edgecolor="white", linewidth=0.7, label=name)
        # Add smooth curve only (no original points)
        x_smooth, y_smooth = smooth_curve(centers, counts)
        # Use shared padded cap for combined panel as well
        local_cap = global_cap
        if local_cap is not None:
            y_smooth = np.minimum(y_smooth, local_cap)
        axc.plot(x_smooth + offs, y_smooth, color=colors[name], linewidth=2.0, linestyle="-", alpha=0.9)
        if len(counts) > 0:
            max_combined = max(max_combined, max(counts))
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
    return bins, baseline_counts, local_counts, proposed_counts


def save_wait_hist_interactive(out_path: Path, city: str, baseline_waits, local_waits=None, proposed_waits=None):
    """Deprecated: HTML generation removed. Only PNG plots are generated now."""
    # HTML generation removed - only PNG plots are generated
    # Maps are the only place where HTML is generated
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpkg", default="data/yuseong_housing_3__point.gpkg")
    p.add_argument("--layer", default="yuseong_housing_2__point")
    p.add_argument("--sample-n", type=int, default=50)
    p.add_argument("--num-dcs", type=int, default=3)
    p.add_argument("--city", default="daejeon")
    p.add_argument("--demand-field", default="A26")
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-distance-objective", action="store_true")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--no-mad", dest="use_mad", action="store_false", help="(debug) use variance for Z3 instead of MAD")
    p.set_defaults(use_mad=True)
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

    # Common base command parts (without --num-dcs, will add --dcs after baseline)
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
    
    # Add --use-distance-objective if specified
    if args.use_distance_objective:
        base_cmd.append("--use-distance-objective")

    # 1) baseline (with fixed/predefined DCs)
    cmd_base = base_cmd.copy()
    # Replace --iters value with 1 for baseline
    iters_idx = cmd_base.index("--iters")
    cmd_base[iters_idx + 1] = "1"
    # Insert DCs before --output-dir (use predefined set when num_dcs==3)
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
        # fallback: let run_experiment generate DCs
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
    
    # Convert depots to --dcs format: "lat,lon,name" or "lat,lon"
    dcs_args = []
    for depot in depots:
        dc_str = f"{depot['lat']},{depot['lon']}"
        if depot.get("name") or depot.get("id"):
            dc_str += f",{depot.get('name') or depot.get('id')}"
        dcs_args.append(dc_str)
    
    logger.info(f"Using DCs from baseline: {len(dcs_args)} DCs")
    for dc_str in dcs_args:
        logger.info(f"  {dc_str}")

    # 2) local (with same DCs as baseline)
    cmd_local = base_cmd.copy()
    # Insert --dcs and DC strings before --output-dir
    output_dir_idx = cmd_local.index("--output-dir")
    cmd_local.insert(output_dir_idx, "--dcs")
    for dc_str in reversed(dcs_args):  # Insert in reverse order to maintain correct order
        cmd_local.insert(output_dir_idx + 1, dc_str)
    cmd_local += ["--method", "local"]
    logger.info("Running local...")
    subprocess.check_call(cmd_local)
    # Load local solution directly from local.json (saved by run_experiment.py)
    local = load_json(solutions_dir / "local.json")
    # Ensure local.json has stops_dict and depots from baseline (needed for scoring/plotting)
    if "stops_dict" not in local:
        local["stops_dict"] = baseline.get("stops_dict", {})
    if "depots" not in local:
        local["depots"] = baseline.get("depots", [])
    # Update local.json with stops_dict/depots if they were missing
    if "stops_dict" not in local or "depots" not in local:
        try:
            (solutions_dir / "local.json").write_text(json.dumps(local, indent=2))
            logger.info(f"Updated local.json with stops_dict/depots")
        except Exception as e:
            logger.warning(f"Failed to update local.json: {e}")

    # 3) proposed (fixed operator) - MAD scoring
    cmd_prop = base_cmd.copy()
    # Insert --dcs and DC strings before --output-dir
    output_dir_idx = cmd_prop.index("--output-dir")
    cmd_prop.insert(output_dir_idx, "--dcs")
    for dc_str in reversed(dcs_args):  # Insert in reverse order to maintain correct order
        cmd_prop.insert(output_dir_idx + 1, dc_str)
    cmd_prop += ["--method", "proposed", "--operator-mode", "fixed"]
    if args.use_mad:
        cmd_prop.append("--use-mad")
    logger.info("Running proposed (ALNS)...")
    subprocess.check_call(cmd_prop)

    # Load proposed (ALNS_MAD.json)
    proposed = load_json(solutions_dir / "ALNS_MAD.json")

    # depots was already extracted from baseline above

    # Create shared OSRM providers for consistent calculation across all solutions
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, baseline.get("stops_dict", {}), cache)
    
    # Always use baseline stops_dict for all solutions (contains metadata like households)
    baseline_stops_dict = baseline.get("stops_dict", {})
    
    # Compute scores using same providers and stops_dict for consistency
    # Baseline: use its own values as normalizers (Z_baseline should be 1.0)
    base_scores = calc_scores(
        baseline, depots, args.use_distance_objective, args.use_mad, args.alpha, args.beta, args.gamma,
        time_provider=time_provider, distance_provider=distance_provider, cache=cache,
        baseline_stops_dict=baseline_stops_dict
    )
    
    # Try to load alns_mad_debug to use exact normalizers from proposed_algorithm
    proposed_debug = None
    debug_file = debug_dir / "alns_mad_debug.json"
    if debug_file.exists():
        try:
            proposed_debug = load_json(debug_file)
            logger.info("Loaded alns_mad_debug.json for normalizers")
        except Exception as e:
            logger.warning(f"Failed to load alns_mad_debug.json: {e}")
    
    # Use normalizers and objectives from alns_mad_debug if available (like compare_Z3_variance_vs_MAD.py)
    if proposed_debug and "normalizers" in proposed_debug:
        normalizers = proposed_debug["normalizers"]
        Z1_star = normalizers.get("Z1_star", base_scores["Z1"])
        Z2_star = normalizers.get("Z2_star", base_scores["Z2"])
        Z3_star = normalizers.get("Z3_star", base_scores["Z3"])
        logger.info(f"Using normalizers from proposed_algorithm: Z1*={Z1_star:.1f}, Z2*={Z2_star:.1f}, Z3*={Z3_star:.1f}")
        
        # Use baseline_objectives from alns_mad_debug for baseline scores (consistency with proposed_algorithm)
        baseline_objectives = proposed_debug.get("baseline_objectives", {})
        if baseline_objectives and all(k in baseline_objectives for k in ["Z1", "Z2", "Z3"]):
            Z1_baseline = baseline_objectives.get("Z1")
            Z2_baseline = baseline_objectives.get("Z2")
            Z3_baseline = baseline_objectives.get("Z3")
            logger.info(f"Using baseline objectives from proposed_algorithm: Z1={Z1_baseline:.1f}, Z2={Z2_baseline:.1f}, Z3={Z3_baseline:.1f}")
            
            # Recompute baseline Z using these values
            from src.vrp_fairness.objectives import compute_combined_Z
            Z_baseline = compute_combined_Z(Z1_baseline, Z2_baseline, Z3_baseline,
                                          Z1_star, Z2_star, Z3_star,
                                          args.alpha, args.beta, args.gamma)
            base_scores = {
                "Z1": Z1_baseline,
                "Z2": Z2_baseline,
                "Z3": Z3_baseline,
                "Z": Z_baseline,
                "waiting": base_scores["waiting"],  # Keep waiting times for plotting
            }
        else:
            logger.warning("baseline_objectives not found in alns_mad_debug, using recomputed baseline scores")
    else:
        Z1_star, Z2_star, Z3_star = base_scores["Z_star"]
        logger.info(f"Using normalizers from baseline: Z1*={Z1_star:.1f}, Z2*={Z2_star:.1f}, Z3*={Z3_star:.1f}")
    
    # Local: use same normalizers and baseline stops_dict
    local_scores = calc_scores(
        local, depots, args.use_distance_objective, args.use_mad, args.alpha, args.beta, args.gamma,
        Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
        time_provider=time_provider, distance_provider=distance_provider, cache=cache,
        baseline_stops_dict=baseline_stops_dict
    )
    
    # Proposed: use best_objectives from alns_mad_debug if available (like compare_Z3_variance_vs_MAD.py)
    best_objectives = None
    if proposed_debug:
        best_objectives = proposed_debug.get("best_objectives", {})
    
    if best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]):
        # Use exact RAW values from proposed_algorithm (computed with same provider and patched compute_Z3)
        Z1_proposed = best_objectives.get("Z1")
        Z2_proposed = best_objectives.get("Z2")
        Z3_proposed = best_objectives.get("Z3")
        
        # Validate values are not None
        if Z1_proposed is None or Z2_proposed is None or Z3_proposed is None:
            logger.warning("best_objectives contains None values! Falling back to recomputation.")
            best_objectives = None
        
        if best_objectives:
            logger.info(f"Using RAW best objectives from proposed_algorithm:")
            logger.info(f"  Z1_proposed (RAW)={Z1_proposed:.2f}")
            logger.info(f"  Z2_proposed (RAW)={Z2_proposed:.2f}")
            logger.info(f"  Z3_proposed (RAW)={Z3_proposed:.2f}")
            
            # Compute normalized Z using same normalizers
            from src.vrp_fairness.objectives import compute_combined_Z
            Z_proposed = compute_combined_Z(Z1_proposed, Z2_proposed, Z3_proposed,
                                           Z1_star, Z2_star, Z3_star,
                                           args.alpha, args.beta, args.gamma)
            
            # Still need waiting times for plotting, so compute those
            prop_scores_temp = calc_scores(
                proposed, depots, args.use_distance_objective, args.use_mad, args.alpha, args.beta, args.gamma,
                Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
                time_provider=time_provider, distance_provider=distance_provider, cache=cache,
                baseline_stops_dict=baseline_stops_dict
            )
            
            prop_scores = {
                "Z1": Z1_proposed,
                "Z2": Z2_proposed,
                "Z3": Z3_proposed,
                "Z": Z_proposed,
                "waiting": prop_scores_temp["waiting"],  # Keep waiting times for plotting
            }
    
    # Fallback: recompute if best_objectives not available
    if not (best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]) and 
            all(best_objectives.get(k) is not None for k in ["Z1", "Z2", "Z3"])):
        logger.warning("best_objectives not found in alns_mad_debug, recomputing proposed scores...")
        prop_scores = calc_scores(
            proposed, depots, args.use_distance_objective, args.use_mad, args.alpha, args.beta, args.gamma,
            Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
            time_provider=time_provider, distance_provider=distance_provider, cache=cache,
            baseline_stops_dict=baseline_stops_dict
        )

    # Z-scores comparison
    rows = [
        ("baseline", base_scores["Z1"], base_scores["Z2"], base_scores["Z3"], base_scores["Z"]),
        ("local", local_scores["Z1"], local_scores["Z2"], local_scores["Z3"], local_scores["Z"]),
        ("ALSM (MAD)", prop_scores["Z1"], prop_scores["Z2"], prop_scores["Z3"], prop_scores["Z"]),
    ]

    with open(data_dir / "baseline_local_alns_mad_scores.csv", "w", newline="") as f:
        import csv

        w = csv.writer(f)
        w.writerow(["method", "Z1", "Z2", "Z3_MAD", "Z"])
        w.writerows(rows)
    logger.info(f"Saved Z-scores: {data_dir / 'baseline_local_alns_mad_scores.csv'}")
    
    # Metrics comparison (similar to run_experiment.py's comparison.csv)
    baseline_metrics = baseline.get("metrics", {})
    local_metrics = local.get("metrics", {})
    proposed_metrics = proposed.get("metrics", {})
    
    with open(data_dir / "baseline_local_alns_mad_metrics.csv", "w", newline="") as f:
        import csv
        
        w = csv.writer(f)
        w.writerow(["Metric", "Baseline", "Local", "ALSM (MAD)", "Local Change %", "ALSM Change %"])
        
        # Compare key metrics
        key_metrics = ["W_max", "W_mean", "W_p95", "W_top10_mean", "total_cost", "driver_balance", "num_vehicles_used"]
        for key in key_metrics:
            if key in baseline_metrics:
                base_val = baseline_metrics[key]
                local_val = local_metrics.get(key, 0)
                prop_val = proposed_metrics.get(key, 0)
                
                if isinstance(base_val, (int, float)) and base_val != 0:
                    local_change = ((local_val - base_val) / base_val) * 100
                    prop_change = ((prop_val - base_val) / base_val) * 100
                else:
                    local_change = 0.0
                    prop_change = 0.0
                
                w.writerow([key, base_val, local_val, prop_val, f"{local_change:.2f}%", f"{prop_change:.2f}%"])
    logger.info(f"Saved metrics comparison: {data_dir / 'baseline_local_alns_mad_metrics.csv'}")

    # Plot waits (weighted exactly like Z1 = max_i households_i * w_i)
    def _weighted_waits(wait_dict: Dict[str, float]) -> List[float]:
        weighted = []
        for stop_id, w in wait_dict.items():
            stop_meta = baseline_stops_dict.get(stop_id, {})
            weight = stop_meta.get("households", stop_meta.get("demand", 1))
            if weight in (None, 0):
                weight = 1
            weighted.append(w * weight)
        return weighted

    # Collect raw waits and their household weights for frequency weighting
    # X-axis: raw waiting time (seconds)
    # Frequencies: weighted by households (or demand if households missing)
    def _raw_waits_with_weights(wait_dict: Dict[str, float]):
        vals, wts = [], []
        for stop_id, w in wait_dict.items():
            stop_meta = baseline_stops_dict.get(stop_id, {})
            wt = stop_meta.get("households", stop_meta.get("demand", 1))
            if wt in (None, 0):
                wt = 1
            vals.append(w)
            wts.append(wt)
        return vals, wts

    baseline_waits, baseline_wts = _raw_waits_with_weights(base_scores["waiting"])
    local_waits, local_wts = _raw_waits_with_weights(local_scores["waiting"])
    proposed_waits, proposed_wts = _raw_waits_with_weights(prop_scores["waiting"])
    
    bins, base_counts, local_counts, prop_counts = plot_wait_panels(
        out_path=plots_dir / "compare_wait_panels.png",
        city=args.city,
        baseline_waits=baseline_waits,
        local_waits=local_waits,
        proposed_waits=proposed_waits,
        baseline_weights=baseline_wts,
        local_weights=local_wts,
        proposed_weights=proposed_wts,
    )
    # HTML generation removed - only PNG plots are generated
    # Maps are the only place where HTML is generated

    # Persist plot-ready data so plots can be regenerated from CSV
    waits_path = data_dir / "baseline_local_alns_mad_wait_values.csv"
    with open(waits_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["method", "waiting_time_seconds", "households"])
        for v, wt in zip(baseline_waits, baseline_wts):
            w.writerow(["baseline", v, wt])
        for v, wt in zip(local_waits, local_wts):
            w.writerow(["local", v, wt])
        for v, wt in zip(proposed_waits, proposed_wts):
            w.writerow(["ALSM (MAD)", v, wt])
    logger.info(f"Saved wait values for plotting: {waits_path}")

    hist_path = data_dir / "baseline_local_alns_mad_wait_hist_data.csv"
    with open(hist_path, "w", newline="") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "bin_center", "baseline_count", "local_count", "ALSM_count"])
        for i in range(len(bins) - 1):
            left = bins[i]
            right = bins[i + 1]
            center = (left + right) / 2
            w.writerow([left, right, center, base_counts[i], local_counts[i], prop_counts[i]])
    logger.info(f"Saved histogram data for plotting: {hist_path}")


if __name__ == "__main__":
    main()

