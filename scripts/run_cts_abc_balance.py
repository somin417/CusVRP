#!/usr/bin/env python3
"""
Run CTS with different alpha/beta/gamma balance configurations.
Beta is fixed at 0.3 (cost), alpha and gamma vary.

Configurations:
- (0.6, 0.3, 0.1): Z1-focused
- (0.35, 0.3, 0.35): Balanced
- (0.1, 0.3, 0.6): Z3-focused

Stores best results in solutions/abc_balance/ and creates comparison plots.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.utils.utils import load_json
from src.vrp_fairness.objectives import (
    compute_waiting_times,
    compute_Z1,
    compute_Z2,
    compute_Z3_MAD,
)
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_cts(
    baseline_path: Path,
    alpha: float,
    beta: float,
    gamma: float,
    iters: int,
    seed: int,
    eps: float,
    output_dir: Path,
    config_name: str,
    gpkg: str = "data/yuseong_housing_3__point.gpkg",
    layer: str = "yuseong_housing_2__point"
) -> Dict[str, Any]:
    """
    Run CTS with given parameters and return best solution.
    
    Returns:
        Dict with solution, scores, and metadata
    """
    logger.info(f"Running CTS with (alpha={alpha}, beta={beta}, gamma={gamma}) - {config_name}")
    
    # Load baseline to get depots
    baseline = load_json(baseline_path)
    depots = baseline.get("depots", [])
    
    if not depots:
        raise ValueError("No depots found in baseline")
    
    # Build DCs argument
    dcs_args = []
    for depot in depots:
        dc_str = f"{depot['lat']},{depot['lon']}"
        if depot.get("name") or depot.get("id"):
            dc_str += f",{depot.get('name') or depot.get('id')}"
        dcs_args.append(dc_str)
    
    # Create temporary output directory for this run
    temp_output = output_dir / f"temp_{config_name}"
    temp_output.mkdir(parents=True, exist_ok=True)
    
    # Use 20 stops for abc_balance experiment
    num_stops = 20
    
    # Build command
    cmd = [
        "python", "-m", "src.vrp_fairness.run_experiment",
        "--seed", str(seed),
        "--gpkg", gpkg,
        "--layer", layer,
        "--sample-n", str(num_stops),
        "--city", "daejeon",
        "--demand-field", "A26",
        "--eps", str(eps),
        "--iters", str(iters),
        "--alpha", str(alpha),
        "--beta", str(beta),
        "--gamma", str(gamma),
        "--output-dir", str(temp_output),
        "--no-plots",
        "--method", "proposed",
        "--operator-mode", "cts",
        "--use-mad",
    ]
    
    # Add DCs
    cmd.extend(["--dcs"] + dcs_args)
    
    # Run CTS
    # Force unbuffered output so progress lines appear immediately in tmux
    env = dict(**os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    try:
        subprocess.check_call(cmd, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"CTS run failed: {e}")
        raise
    
    # Load results
    cts_solution_file = temp_output / "cts_solution.json"
    cts_debug_file = temp_output / "cts_debug.json"
    
    if not cts_solution_file.exists():
        raise FileNotFoundError(f"CTS solution not found: {cts_solution_file}")
    
    solution = load_json(cts_solution_file)
    debug = load_json(cts_debug_file) if cts_debug_file.exists() else {}
    
    # Extract best solution from debug if available
    best_backup = debug.get("best_solution_backup")
    if best_backup:
        solution = best_backup.get("solution", solution)
        best_Z = best_backup.get("Z")
        best_Z1 = best_backup.get("Z1")
        best_Z2 = best_backup.get("Z2")
        best_Z3 = best_backup.get("Z3")
        best_iter = best_backup.get("iteration")
    else:
        # Compute scores from solution
        stops_by_id = baseline.get("stops_dict", {})
        cache = iNaviCache(approx_mode=False)
        time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
        waiting = compute_waiting_times(solution, stops_by_id, time_provider)
        best_Z1 = compute_Z1(waiting, stops_by_id)
        best_Z2 = compute_Z2(solution, distance_provider, time_provider, False)
        best_Z3 = compute_Z3_MAD(waiting, stops_by_id)
        
        # Get baseline normalizers
        baseline_waiting = compute_waiting_times(baseline, stops_by_id, time_provider)
        baseline_Z1 = compute_Z1(baseline_waiting, stops_by_id)
        baseline_Z2 = compute_Z2(baseline, distance_provider, time_provider, False)
        baseline_Z3 = compute_Z3_MAD(baseline_waiting, stops_by_id)
        
        from src.vrp_fairness.objectives import compute_combined_Z
        best_Z = compute_combined_Z(best_Z1, best_Z2, best_Z3, 
                                    baseline_Z1, baseline_Z2, baseline_Z3,
                                    alpha, beta, gamma)
        best_iter = None
    
    result = {
        "solution": solution,
        "config_name": config_name,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "Z": best_Z,
        "Z1": best_Z1,
        "Z2": best_Z2,
        "Z3": best_Z3,
        "best_iteration": best_iter,
        "iters": iters,
        "seed": seed,
        "eps": eps
    }
    
    logger.info(f"  Completed: Z={best_Z:.4f}, Z1={best_Z1:.1f}, Z2={best_Z2:.1f}, Z3={best_Z3:.1f}")
    
    return result


def save_best_solution(result: Dict[str, Any], output_dir: Path, force: bool = False):
    """Save best solution to solutions/abc_balance/"""
    best_dir = output_dir / "solutions" / "abc_balance"
    best_dir.mkdir(parents=True, exist_ok=True)
    
    config_name = result["config_name"]
    filename = f"cts_{config_name}_best.json"
    filepath = best_dir / filename
    
    if filepath.exists() and not force:
        logger.warning(f"Best solution file already exists: {filepath}")
        logger.warning("Use --force to overwrite, or the file will be skipped")
        return filepath
    
    # Prepare data to save
    save_data = {
        "solution": result["solution"],
        "config_name": config_name,
        "alpha": result["alpha"],
        "beta": result["beta"],
        "gamma": result["gamma"],
        "Z": result["Z"],
        "Z1": result["Z1"],
        "Z2": result["Z2"],
        "Z3": result["Z3"],
        "best_iteration": result["best_iteration"],
        "iters": result["iters"],
        "seed": result["seed"],
        "eps": result["eps"]
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    logger.info(f"Saved best solution: {filepath}")
    return filepath


def plot_comparison(results: List[Dict[str, Any]], output_path: Path):
    """Create four comparison plots: raw and normalized Z scores per balance and comparison."""
    config_names = [r["config_name"] for r in results]
    z1_values = [r["Z1"] for r in results]
    z2_values = [r["Z2"] for r in results]
    z3_values = [r["Z3"] for r in results]
    z_values = [r["Z"] for r in results]
    
    # Normalize values to 0-1 range (divide by max)
    max_z1 = max(z1_values) if z1_values else 1.0
    max_z2 = max(z2_values) if z2_values else 1.0
    max_z3 = max(z3_values) if z3_values else 1.0
    max_z = max(z_values) if z_values else 1.0
    
    z1_norm = [z / max_z1 for z in z1_values]
    z2_norm = [z / max_z2 for z in z2_values]
    z3_norm = [z / max_z3 for z in z3_values]
    z_norm = [z / max_z for z in z_values]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 모델별 색상: 1, 3번 그래프와 비슷한 계열
    model_colors = ['#4C78A8', '#54A24B', '#F58518']  # z1_focused, balanced, z3_focused
    
    # 각 모델별 색상 계열 (Z1, Z2, Z3, Z* 순서)
    # z1_focused: 파란색 계열
    z1_focused_colors = ['#2E5C8A', '#3D6B9E', '#4C78A8', '#5B85B2']
    # balanced: 초록색 계열
    balanced_colors = ['#3A7A3F', '#46964D', '#54A24B', '#62AE59']
    # z3_focused: 주황색 계열
    z3_focused_colors = ['#D66B0E', '#E47813', '#F58518', '#FF9223']
    
    model_color_palettes = [z1_focused_colors, balanced_colors, z3_focused_colors]
    
    x_pos = np.arange(len(config_names))
    width = 0.2
    
    # Plot 1: Raw Z scores per balance (x=3개 모델, y=z별 score 3개: Z1, Z2, Z3)
    ax1 = axes[0, 0]
    for i, config_name in enumerate(config_names):
        colors = model_color_palettes[i]
        ax1.bar(x_pos[i] - width, z1_values[i], width, color=colors[0], alpha=0.8)
        ax1.bar(x_pos[i], z2_values[i], width, color=colors[1], alpha=0.8)
        ax1.bar(x_pos[i] + width, z3_values[i], width, color=colors[2], alpha=0.8)
    # 범례는 모델별로만 표시
    legend_elements = [Patch(facecolor=model_colors[i], alpha=0.8, label=config_names[i]) for i in range(len(config_names))]
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Z Scores per Balance (Raw)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(config_names, rotation=0, ha='center')
    ax1.legend(handles=legend_elements)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Raw Z scores comparison (x=Z1, Z2, Z3, y=3개 모델 값들이 붙어있는 구조)
    ax2 = axes[0, 1]
    z_labels = ['Z1', 'Z2', 'Z3']
    x_pos2 = np.arange(len(z_labels))
    width2 = 0.25
    
    for i, (label, data) in enumerate(zip(config_names, zip(z1_values, z2_values, z3_values))):
        offset = (i - 1) * width2
        ax2.bar(x_pos2 + offset, data, width2, label=label, color=model_colors[i], alpha=0.8)
    
    ax2.set_xlabel('Z Score Type', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Z Scores Comparison (Raw)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(z_labels, rotation=0, ha='center')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Normalized Z scores per balance (x=3개 모델, y=z별 score 4개: Z1*, Z2*, Z3*, Z*)
    ax3 = axes[1, 0]
    for i, config_name in enumerate(config_names):
        colors = model_color_palettes[i]
        ax3.bar(x_pos[i] - 1.5*width, z1_norm[i], width, color=colors[0], alpha=0.8)
        ax3.bar(x_pos[i] - 0.5*width, z2_norm[i], width, color=colors[1], alpha=0.8)
        ax3.bar(x_pos[i] + 0.5*width, z3_norm[i], width, color=colors[2], alpha=0.8)
        ax3.bar(x_pos[i] + 1.5*width, z_norm[i], width, color=colors[3], alpha=0.8)
    # 범례는 모델별로만 표시
    legend_elements3 = [Patch(facecolor=model_colors[i], alpha=0.8, label=config_names[i]) for i in range(len(config_names))]
    ax3.set_xlabel('Model', fontsize=12)
    ax3.set_ylabel('Normalized Score', fontsize=12)
    ax3.set_title('Z Scores per Balance (Normalized)', fontsize=14, fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(config_names, rotation=0, ha='center')
    ax3.set_ylim([0, 1.1])
    ax3.legend(handles=legend_elements3)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Normalized Z scores comparison (x=Z1*, Z2*, Z3*, Z*, y=3개 모델 값들이 붙어있는 구조)
    ax4 = axes[1, 1]
    z_labels_norm = ['Z1*', 'Z2*', 'Z3*', 'Z*']
    x_pos4 = np.arange(len(z_labels_norm))
    
    for i, (label, data) in enumerate(zip(config_names, zip(z1_norm, z2_norm, z3_norm, z_norm))):
        offset = (i - 1) * width2
        ax4.bar(x_pos4 + offset, data, width2, label=label, color=model_colors[i], alpha=0.8)
    
    ax4.set_xlabel('Z Score Type', fontsize=12)
    ax4.set_ylabel('Normalized Score', fontsize=12)
    ax4.set_title('Z Scores Comparison (Normalized)', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos4)
    ax4.set_xticklabels(z_labels_norm, rotation=0, ha='center')
    ax4.set_ylim([0, 1.1])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved comparison plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run CTS with different alpha/beta/gamma balance")
    parser.add_argument("--baseline", type=str, default="outputs/solutions/baseline.json",
                       help="Path to baseline.json")
    parser.add_argument("--iters", type=int, default=20,
                       help="Number of iterations (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--eps", type=float, default=0.30,
                       help="Cost budget tolerance (default: 0.30 = 130% of baseline Z2). Note: Higher than main experiments (0.10) to allow more exploration for ABC balance comparison.")
    parser.add_argument("--output-dir", type=str, default="outputs",
                       help="Output directory (default: outputs)")
    parser.add_argument("--gpkg", type=str, default="data/yuseong_housing_3__point.gpkg",
                       help="GPKG file path (default: data/yuseong_housing_3__point.gpkg)")
    parser.add_argument("--layer", type=str, default="yuseong_housing_2__point",
                       help="Layer name (default: yuseong_housing_2__point)")
    parser.add_argument("--force", action="store_true", help="Force overwrite of existing solution files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create organized subdirectories
    solutions_dir = output_dir / "solutions"
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    solutions_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    baseline_path = Path(args.baseline)
    # Check in solutions directory if not found at specified path
    if not baseline_path.exists() and (solutions_dir / "baseline.json").exists():
        baseline_path = solutions_dir / "baseline.json"
    if not baseline_path.exists():
        logger.error(f"Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    # Define configurations: (alpha, beta, gamma, name)
    configurations = [
        (0.6, 0.3, 0.1, "z1_focused"),
        (0.35, 0.3, 0.35, "balanced"),
        (0.1, 0.3, 0.6, "z3_focused"),
    ]
    
    results = []
    
    # Run CTS for each configuration
    for alpha, beta, gamma, config_name in configurations:
        try:
            result = run_cts(
                baseline_path=baseline_path,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                iters=args.iters,
                seed=args.seed,
                eps=args.eps,
                output_dir=output_dir,
                config_name=config_name,
                gpkg=args.gpkg,
                layer=args.layer
            )
            results.append(result)
            
            # Save best solution
            save_best_solution(result, output_dir, force=args.force)
            
        except Exception as e:
            logger.error(f"Failed to run CTS for {config_name}: {e}")
            continue
    
    if not results:
        logger.error("No successful runs!")
        sys.exit(1)
    
    # Create comparison plot
    plot_path = plots_dir / "abc_balance_comparison.png"
    plot_comparison(results, plot_path)
    
    # Save summary JSON
    summary = {
        "configurations": [
            {
                "name": r["config_name"],
                "alpha": r["alpha"],
                "beta": r["beta"],
                "gamma": r["gamma"],
                "Z": r["Z"],
                "Z1": r["Z1"],
                "Z2": r["Z2"],
                "Z3": r["Z3"],
                "best_iteration": r["best_iteration"]
            }
            for r in results
        ],
        "iters": args.iters,
        "seed": args.seed,
        "eps": args.eps
    }
    
    summary_path = data_dir / "abc_balance_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("CTS ABC BALANCE COMPARISON")
    print("=" * 80)
    print(f"{'Config':<15} {'α':<6} {'β':<6} {'γ':<6} {'Z':<10} {'Z1':<12} {'Z2':<12} {'Z3':<12}")
    print("-" * 80)
    for r in results:
        print(f"{r['config_name']:<15} {r['alpha']:<6.2f} {r['beta']:<6.2f} {r['gamma']:<6.2f} "
              f"{r['Z']:<10.4f} {r['Z1']:<12.1f} {r['Z2']:<12.1f} {r['Z3']:<12.1f}")
    print("=" * 80)
    print(f"\nBest solutions saved to: {output_dir / 'solutions' / 'abc_balance'}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
