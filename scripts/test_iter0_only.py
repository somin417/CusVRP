#!/usr/bin/env python3
"""
일회용 스크립트: compare_waiting_and_scores.py와 동일하되,
improved (proposed)는 iteration 0만 실행하여 solution과 값들을 확인.

Usage:
    python scripts/test_iter0_only.py --sample-n 50 --iters 50
"""

import argparse
import csv
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
    """Calculate Z-scores for a solution using MAD for Z3 (spec)."""
    if cache is None:
        cache = iNaviCache(approx_mode=False)
    
    if baseline_stops_dict is None:
        baseline_stops_dict = solution.get("stops_dict", {})
    stops_by_id = baseline_stops_dict
    
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


def main():
    p = argparse.ArgumentParser(description="Test iter 0 only for proposed algorithm")
    p.add_argument("--gpkg", default="data/yuseong_housing_3__point.gpkg")
    p.add_argument("--layer", default="yuseong_housing_2__point")
    p.add_argument("--sample-n", type=int, default=50)
    p.add_argument("--num-dcs", type=int, default=3)
    p.add_argument("--city", default="daejeon")
    p.add_argument("--demand-field", default="A26")
    p.add_argument("--eps", type=float, default=0.10)
    p.add_argument("--iters", type=int, default=50, help="Total iterations (but proposed will use 1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-distance-objective", action="store_true")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.3)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--output-dir", default="outputs")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(exist_ok=True)

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

    # 1) baseline
    cmd_base = base_cmd.copy()
    iters_idx = cmd_base.index("--eps") + 2  # Find position after --eps
    cmd_base.insert(iters_idx, "--iters")
    cmd_base.insert(iters_idx + 1, "1")
    
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
    baseline = load_json(out / "baseline.json")
    depots = baseline.get("depots", [])
    if not depots:
        raise RuntimeError("No depots found in baseline.json")
    
    dcs_args = []
    for depot in depots:
        dc_str = f"{depot['lat']},{depot['lon']}"
        if depot.get("name") or depot.get("id"):
            dc_str += f",{depot.get('name') or depot.get('id')}"
        dcs_args.append(dc_str)
    
    logger.info(f"Using DCs from baseline: {len(dcs_args)} DCs")

    # 2) local
    cmd_local = base_cmd.copy()
    output_dir_idx = cmd_local.index("--output-dir")
    cmd_local.insert(output_dir_idx, "--dcs")
    for dc_str in reversed(dcs_args):
        cmd_local.insert(output_dir_idx + 1, dc_str)
    iters_idx = cmd_local.index("--eps") + 2
    cmd_local.insert(iters_idx, "--iters")
    cmd_local.insert(iters_idx + 1, str(args.iters))  # local은 원래 iters 유지
    cmd_local += ["--method", "local"]
    logger.info("Running local...")
    subprocess.check_call(cmd_local)
    local = load_json(out / "local.json")
    if "stops_dict" not in local:
        local["stops_dict"] = baseline.get("stops_dict", {})
    if "depots" not in local:
        local["depots"] = baseline.get("depots", [])

    # 3) proposed (FIXED: iter 0만 실행 = --iters 1)
    cmd_prop = base_cmd.copy()
    output_dir_idx = cmd_prop.index("--output-dir")
    cmd_prop.insert(output_dir_idx, "--dcs")
    for dc_str in reversed(dcs_args):
        cmd_prop.insert(output_dir_idx + 1, dc_str)
    iters_idx = cmd_prop.index("--eps") + 2
    cmd_prop.insert(iters_idx, "--iters")
    cmd_prop.insert(iters_idx + 1, "1")  # proposed는 iter 0만 (1 iteration)
    cmd_prop += ["--method", "proposed", "--operator-mode", "fixed", "--use-mad"]
    logger.info("Running proposed (ALNS) with --iters 1 (iter 0 only)...")
    subprocess.check_call(cmd_prop)

    # Load proposed
    proposed = load_json(out / "improved.json")
    proposed_debug = None
    debug_file = out / "proposed_debug.json"
    if debug_file.exists():
        proposed_debug = load_json(debug_file)
        logger.info("Loaded proposed_debug.json")

    # Create shared OSRM providers
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, baseline.get("stops_dict", {}), cache)
    baseline_stops_dict = baseline.get("stops_dict", {})
    
    # Compute scores
    base_scores = calc_scores(
        baseline, depots, args.use_distance_objective, True, args.alpha, args.beta, args.gamma,
        time_provider=time_provider, distance_provider=distance_provider, cache=cache,
        baseline_stops_dict=baseline_stops_dict
    )
    
    # Use normalizers from proposed_debug if available
    if proposed_debug and "normalizers" in proposed_debug:
        normalizers = proposed_debug["normalizers"]
        Z1_star = normalizers.get("Z1_star", base_scores["Z1"])
        Z2_star = normalizers.get("Z2_star", base_scores["Z2"])
        Z3_star = normalizers.get("Z3_star", base_scores["Z3"])
        logger.info(f"Using normalizers from proposed_algorithm: Z1*={Z1_star:.1f}, Z2*={Z2_star:.1f}, Z3*={Z3_star:.1f}")
        
        baseline_objectives = proposed_debug.get("baseline_objectives", {})
        if baseline_objectives and all(k in baseline_objectives for k in ["Z1", "Z2", "Z3"]):
            Z1_baseline = baseline_objectives.get("Z1")
            Z2_baseline = baseline_objectives.get("Z2")
            Z3_baseline = baseline_objectives.get("Z3")
            logger.info(f"Using baseline objectives from proposed_algorithm: Z1={Z1_baseline:.1f}, Z2={Z2_baseline:.1f}, Z3={Z3_baseline:.1f}")
            
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
    
    local_scores = calc_scores(
        local, depots, args.use_distance_objective, True, args.alpha, args.beta, args.gamma,
        Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
        time_provider=time_provider, distance_provider=distance_provider, cache=cache,
        baseline_stops_dict=baseline_stops_dict
    )
    
    # Proposed: use best_objectives from proposed_debug if available
    best_objectives = None
    if proposed_debug:
        best_objectives = proposed_debug.get("best_objectives", {})
    
    if best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]):
        Z1_proposed = best_objectives.get("Z1")
        Z2_proposed = best_objectives.get("Z2")
        Z3_proposed = best_objectives.get("Z3")
        
        if Z1_proposed is None or Z2_proposed is None or Z3_proposed is None:
            logger.warning("best_objectives contains None values! Falling back to recomputation.")
            best_objectives = None
        
        if best_objectives:
            logger.info(f"Using RAW best objectives from proposed_algorithm:")
            logger.info(f"  Z1_proposed (RAW)={Z1_proposed:.2f}")
            logger.info(f"  Z2_proposed (RAW)={Z2_proposed:.2f}")
            logger.info(f"  Z3_proposed (RAW)={Z3_proposed:.2f}")
            
            from src.vrp_fairness.objectives import compute_combined_Z
            Z_proposed = compute_combined_Z(Z1_proposed, Z2_proposed, Z3_proposed,
                                           Z1_star, Z2_star, Z3_star,
                                           args.alpha, args.beta, args.gamma)
            
            prop_scores_temp = calc_scores(
                proposed, depots, args.use_distance_objective, True, args.alpha, args.beta, args.gamma,
                Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
                time_provider=time_provider, distance_provider=distance_provider, cache=cache,
                baseline_stops_dict=baseline_stops_dict
            )
            
            prop_scores = {
                "Z1": Z1_proposed,
                "Z2": Z2_proposed,
                "Z3": Z3_proposed,
                "Z": Z_proposed,
                "waiting": prop_scores_temp["waiting"],
            }
    
    if not (best_objectives and all(k in best_objectives for k in ["Z1", "Z2", "Z3"]) and 
            all(best_objectives.get(k) is not None for k in ["Z1", "Z2", "Z3"])):
        logger.warning("best_objectives not found in proposed_debug, recomputing proposed scores...")
        prop_scores = calc_scores(
            proposed, depots, args.use_distance_objective, True, args.alpha, args.beta, args.gamma,
            Z1_star=Z1_star, Z2_star=Z2_star, Z3_star=Z3_star,
            time_provider=time_provider, distance_provider=distance_provider, cache=cache,
            baseline_stops_dict=baseline_stops_dict
        )

    # Override with best_solutions backup if available (preserve waiting/Z1/Z2/Z3 from run time)
    best_dir = out / "best_solutions"
    best_files = sorted(best_dir.glob("*_best.json"), key=lambda p: p.stat().st_mtime) if best_dir.exists() else []
    if best_files:
        best_file = best_files[-1]
        try:
            best_data = load_json(best_file)
            if all(k in best_data for k in ["waiting", "Z1", "Z2", "Z3"]):
                prop_scores = {
                    "Z1": best_data["Z1"],
                    "Z2": best_data["Z2"],
                    "Z3": best_data["Z3"],
                    "Z": best_data.get("Z") or compute_combined_Z(
                        best_data["Z1"], best_data["Z2"], best_data["Z3"],
                        Z1_star, Z2_star, Z3_star,
                        args.alpha, args.beta, args.gamma
                    ),
                    "waiting": best_data["waiting"],
                }
                logger.info(f"Using best_solutions backup: {best_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load best backup {best_file}: {e}")

    # --- Plot waits (same 로직 as compare_waiting_and_scores.py) ---
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

    # Plot (same 파일명/포맷)
    from compare_waiting_and_scores import plot_wait_panels, save_wait_hist_interactive  # reuse existing plotting code
    bins, base_counts, local_counts, prop_counts = plot_wait_panels(
        out_path=out / "compare_wait_panels.png",
        city=args.city,
        baseline_waits=baseline_waits,
        local_waits=local_waits,
        proposed_waits=proposed_waits,
        baseline_weights=baseline_wts,
        local_weights=local_wts,
        proposed_weights=proposed_wts,
    )
    save_wait_hist_interactive(
        out_path=out / "compare_wait_panels.html",
        city=args.city,
        baseline_waits=baseline_waits,
        local_waits=local_waits,
        proposed_waits=proposed_waits,
    )

    # Persist plot-ready data (same 파일명)
    waits_path = out / "compare_wait_values.csv"
    with open(waits_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "waiting_time_seconds", "households"])
        for v, wt in zip(baseline_waits, baseline_wts):
            w.writerow(["baseline", v, wt])
        for v, wt in zip(local_waits, local_wts):
            w.writerow(["local", v, wt])
        for v, wt in zip(proposed_waits, proposed_wts):
            w.writerow(["ALSM (MAD)", v, wt])

    hist_path = out / "compare_wait_hist_data.csv"
    with open(hist_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin_left", "bin_right", "bin_center", "baseline_count", "local_count", "ALSM_count"])
        for i in range(len(bins) - 1):
            left = bins[i]
            right = bins[i + 1]
            center = (left + right) / 2
            w.writerow([left, right, center, base_counts[i], local_counts[i], prop_counts[i]])

    # Print detailed results
    print("\n" + "=" * 80)
    print("ITER 0 ONLY TEST RESULTS")
    print("=" * 80)
    
    print(f"\nBaseline:")
    print(f"  Z  = {base_scores['Z']:.6f}")
    print(f"  Z1 = {base_scores['Z1']:.1f}")
    print(f"  Z2 = {base_scores['Z2']:.1f}")
    print(f"  Z3 = {base_scores['Z3']:.1f}")
    
    print(f"\nLocal:")
    print(f"  Z  = {local_scores['Z']:.6f}")
    print(f"  Z1 = {local_scores['Z1']:.1f}")
    print(f"  Z2 = {local_scores['Z2']:.1f}")
    print(f"  Z3 = {local_scores['Z3']:.1f}")
    
    print(f"\nProposed (Iter 0 only):")
    print(f"  Z  = {prop_scores['Z']:.6f}")
    print(f"  Z1 = {prop_scores['Z1']:.1f}")
    print(f"  Z2 = {prop_scores['Z2']:.1f}")
    print(f"  Z3 = {prop_scores['Z3']:.1f}")
    
    # Check trace
    if proposed_debug and "trace" in proposed_debug:
        trace = proposed_debug["trace"]
        if trace:
            iter0 = trace[0]
            print(f"\nTrace Iter 0:")
            print(f"  Z  = {iter0.get('Z', 'N/A')}")
            print(f"  Z1 = {iter0.get('Z1', 'N/A')}")
            print(f"  Z2 = {iter0.get('Z2', 'N/A')}")
            print(f"  Z3 = {iter0.get('Z3', 'N/A')}")
            print(f"  Accepted = {iter0.get('accepted', 'N/A')}")
            
            # Compare
            if isinstance(iter0.get('Z1'), (int, float)):
                trace_Z1 = iter0['Z1']
                final_Z1 = prop_scores['Z1']
                if abs(trace_Z1 - final_Z1) < 0.1:
                    print(f"\n✓ Trace Z1 ({trace_Z1:.1f}) matches final Z1 ({final_Z1:.1f})")
                else:
                    print(f"\n✗ DISCREPANCY: Trace Z1 ({trace_Z1:.1f}) != final Z1 ({final_Z1:.1f})")
                    print(f"  Difference: {abs(trace_Z1 - final_Z1):.1f}")
    
    # Check best_iteration
    if proposed_debug:
        best_iter = proposed_debug.get("best_iteration", -1)
        best_Z_tracked = proposed_debug.get("best_Z_tracked")
        print(f"\nBest iteration from debug: {best_iter}")
        if best_Z_tracked:
            print(f"Best Z tracked: {best_Z_tracked:.6f}")
            if abs(prop_scores['Z'] - best_Z_tracked) < 0.001:
                print(f"✓ Final Z matches tracked best_Z")
            else:
                print(f"✗ DISCREPANCY: Final Z ({prop_scores['Z']:.6f}) != tracked best_Z ({best_Z_tracked:.6f})")
    
    # Save solution details
    solution_details = {
        "baseline": {
            "Z": base_scores['Z'],
            "Z1": base_scores['Z1'],
            "Z2": base_scores['Z2'],
            "Z3": base_scores['Z3'],
        },
        "local": {
            "Z": local_scores['Z'],
            "Z1": local_scores['Z1'],
            "Z2": local_scores['Z2'],
            "Z3": local_scores['Z3'],
        },
        "proposed_iter0": {
            "Z": prop_scores['Z'],
            "Z1": prop_scores['Z1'],
            "Z2": prop_scores['Z2'],
            "Z3": prop_scores['Z3'],
        },
        "trace_iter0": trace[0] if proposed_debug and proposed_debug.get("trace") else None,
        "best_iteration": proposed_debug.get("best_iteration", -1) if proposed_debug else -1,
    }
    
    details_file = out / "iter0_test_results.json"
    with open(details_file, 'w') as f:
        json.dump(solution_details, f, indent=2)
    logger.info(f"Saved detailed results to {details_file}")
    
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {details_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
