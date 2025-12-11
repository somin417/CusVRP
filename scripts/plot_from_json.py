#!/usr/bin/env python3
"""
Generate plots from existing JSON solution files.
Loads baseline.json and improved.json (or proposed solution) and generates waiting time histograms.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vrp_fairness.objectives import (
    compute_waiting_times, compute_Z1, compute_Z2, compute_Z3, compute_combined_Z
)
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache
from src.vrp_fairness.plotting import plot_waiting_time_histograms

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def compute_scores_from_json(
    solution: Dict[str, Any],
    depots: List[Dict[str, Any]],
    use_distance: bool = False,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    baseline_normalizers: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Compute Z1, Z2, Z3, Z from solution JSON.
    
    Args:
        solution: Solution dictionary
        depots: List of depot dictionaries
        use_distance: Use distance for Z2
        alpha, beta, gamma: Weights for combined Z
        baseline_normalizers: Dict with Z1_star, Z2_star, Z3_star (if None, uses solution's own values)
    
    Returns:
        Dict with Z1, Z2, Z3, Z, waiting
    """
    cache = iNaviCache(approx_mode=False)
    stops_by_id = solution.get("stops_dict", {})
    
    if not stops_by_id:
        logger.warning("stops_dict not found in solution, cannot compute scores")
        return {"Z1": 0.0, "Z2": 0.0, "Z3": 0.0, "Z": 0.0, "waiting": {}}
    
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    waiting = compute_waiting_times(solution, stops_by_id, time_provider)
    
    Z1 = compute_Z1(waiting, stops_by_id)
    Z2 = compute_Z2(solution, distance_provider, time_provider, use_distance)
    Z3 = compute_Z3(waiting, stops_by_id)
    
    # Use baseline normalizers if provided, otherwise use solution's own values (for baseline itself)
    if baseline_normalizers:
        Z1_star = baseline_normalizers.get("Z1", Z1)
        Z2_star = baseline_normalizers.get("Z2", Z2)
        Z3_star = baseline_normalizers.get("Z3", Z3)
    else:
        # For baseline solution, normalize by itself (result will be 1.0)
        Z1_star, Z2_star, Z3_star = Z1, Z2, Z3
    
    Z = compute_combined_Z(Z1, Z2, Z3, Z1_star, Z2_star, Z3_star, alpha, beta, gamma)
    
    return {
        "Z1": Z1,
        "Z2": Z2,
        "Z3": Z3,
        "Z": Z,
        "waiting": waiting
    }


def plot_baseline_vs_alns(
    baseline_waits: List[float],
    alns_waits: List[float],
    city_name: str,
    output_path: Path
) -> None:
    """Plot waiting time histogram: Baseline vs ALNS."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    if not baseline_waits:
        ax.text(0.5, 0.5, 'No waiting time data available',
                transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title(f'Waiting Time Distribution ({city_name.title()})', fontsize=14)
        plt.savefig(str(output_path) + ".png", format="png", dpi=150, bbox_inches='tight')
        plt.close()
        return
    
    # Determine bins
    all_waits = baseline_waits + alns_waits
    max_wait = max(all_waits) if all_waits else 1.0
    bins = np.linspace(0, max_wait * 1.05, min(60, max(20, len(baseline_waits) // 3)))
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
    centers = bins[:-1] + bin_width / 2
    
    base_counts, _ = np.histogram(baseline_waits, bins=bins)
    alns_counts, _ = np.histogram(alns_waits, bins=bins)
    
    # Side-by-side bars
    offset = bin_width * 0.25
    ax.bar(
        centers - offset,
        base_counts,
        width=bin_width * 0.4,
        alpha=0.75,
        color='#4C78A8',
        label='Baseline',
        edgecolor='white',
        linewidth=0.7
    )
    ax.bar(
        centers + offset,
        alns_counts,
        width=bin_width * 0.4,
        alpha=0.75,
        color='#F58518',
        label='ALNS',
        edgecolor='white',
        linewidth=0.7
    )
    
    # Statistics text
    stats_lines = []
    if baseline_waits:
        baseline_mean = np.mean(baseline_waits)
        baseline_max = np.max(baseline_waits)
        stats_lines.append(f'Baseline: Mean={baseline_mean:.1f}s, Max={baseline_max:.1f}s')
    
    if alns_waits:
        alns_mean = np.mean(alns_waits)
        alns_max = np.max(alns_waits)
        alns_improvement = ((baseline_max - alns_max) / baseline_max * 100) if baseline_max > 0 else 0
        stats_lines.append(f'ALNS: Mean={alns_mean:.1f}s, Max={alns_max:.1f}s ({alns_improvement:+.1f}%)')
    
    if stats_lines:
        fig.text(
            0.01, 0.99,
            "\n".join(stats_lines),
            ha='left', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85),
            fontsize=10, family='monospace'
        )
    
    # Formatting
    ax.set_xlabel('Weighted Waiting Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'Waiting Time Distribution: Baseline vs ALNS ({city_name.title()})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=11)
    
    # Optional y-limit
    all_counts = np.concatenate([base_counts, alns_counts])
    if all_counts.size > 0:
        median_c = np.median(all_counts)
        p95_c = np.percentile(all_counts, 95)
        max_c = all_counts.max()
        if median_c > 0 and max_c > 5 * median_c:
            ax.set_ylim(top=1.2 * max(p95_c, median_c))
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # Save PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + ".png", format="png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved waiting histogram: {output_path}.png")
    
    # Try interactive HTML
    try:
        import plotly.graph_objects as go
        
        fig_plotly = go.Figure()
        
        fig_plotly.add_trace(go.Histogram(
            x=baseline_waits,
            name='Baseline',
            opacity=0.75,
            marker_color='#4C78A8',
            nbinsx=min(60, max(20, len(baseline_waits) // 3))
        ))
        
        fig_plotly.add_trace(go.Histogram(
            x=alns_waits,
            name='ALNS',
            opacity=0.75,
            marker_color='#F58518',
            nbinsx=min(60, max(20, len(alns_waits) // 3))
        ))
        
        fig_plotly.update_layout(
            title=f'Waiting Time Distribution: Baseline vs ALNS ({city_name.title()})',
            xaxis_title='Weighted Waiting Time (seconds)',
            yaxis_title='Frequency',
            barmode='overlay',
            hovermode='x unified',
            width=1200,
            height=600
        )
        
        fig_plotly.write_html(str(output_path) + ".html")
        logger.info(f"Saved interactive histogram: {output_path}.html")
    except ImportError:
        logger.info("Plotly not available, skipping interactive HTML")


def main():
    parser = argparse.ArgumentParser(description="Generate plots from JSON solution files")
    parser.add_argument("--baseline", type=str, default="outputs/baseline.json",
                        help="Path to baseline.json")
    parser.add_argument("--improved", type=str, default="outputs/improved.json",
                        help="Path to improved.json (or proposed solution)")
    parser.add_argument("--output-dir", type=str, default="outputs/plots",
                        help="Output directory for plots")
    parser.add_argument("--city", type=str, default="daejeon", help="City name")
    parser.add_argument("--use-distance-objective", action="store_true",
                        help="Use distance for Z2")
    parser.add_argument("--alpha", type=float, default=0.5, help="Z1 weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Z2 weight")
    parser.add_argument("--gamma", type=float, default=0.2, help="Z3 weight")
    
    args = parser.parse_args()
    
    # Load JSON files
    baseline_path = Path(args.baseline)
    improved_path = Path(args.improved)
    
    if not baseline_path.exists():
        logger.error(f"Baseline file not found: {baseline_path}")
        return
    
    if not improved_path.exists():
        logger.error(f"Improved file not found: {improved_path}")
        return
    
    logger.info(f"Loading baseline from: {baseline_path}")
    baseline = load_json(baseline_path)
    
    logger.info(f"Loading improved from: {improved_path}")
    improved = load_json(improved_path)
    
    # Extract depots
    depots = []
    if "depots" in baseline:
        depots = baseline["depots"]
    elif "routes_by_dc" in baseline:
        # Try to infer from routes
        for dc_id in baseline["routes_by_dc"].keys():
            # This is a fallback - ideally depots should be in JSON
            depots.append({"id": dc_id, "lat": 0.0, "lon": 0.0})
    
    if not depots:
        logger.warning("No depots found, using default")
        depots = [{"id": "DC1", "lat": 36.3, "lon": 127.3}]
    
    # Compute baseline scores first (normalized by itself, so Z=1.0)
    logger.info("Computing baseline scores...")
    baseline_scores = compute_scores_from_json(
        baseline, depots, args.use_distance_objective, args.alpha, args.beta, args.gamma,
        baseline_normalizers=None  # Baseline normalizes by itself
    )
    
    # Extract baseline normalizers
    baseline_normalizers = {
        "Z1": baseline_scores["Z1"],
        "Z2": baseline_scores["Z2"],
        "Z3": baseline_scores["Z3"]
    }
    
    # Compute improved scores using baseline normalizers
    logger.info("Computing improved scores...")
    improved_scores = compute_scores_from_json(
        improved, depots, args.use_distance_objective, args.alpha, args.beta, args.gamma,
        baseline_normalizers=baseline_normalizers  # Use baseline normalizers
    )
    
    # Print Z scores
    print("\n" + "=" * 80)
    print("Z SCORES COMPARISON")
    print("=" * 80)
    print(f"{'Metric':<10} {'Baseline':<15} {'ALNS':<15} {'Change %':<15}")
    print("-" * 80)
    
    for metric in ["Z", "Z1", "Z2", "Z3"]:
        base_val = baseline_scores.get(metric, 0.0)
        imp_val = improved_scores.get(metric, 0.0)
        if base_val > 0:
            change_pct = ((base_val - imp_val) / base_val) * 100
        else:
            change_pct = 0.0
        print(f"{metric:<10} {base_val:<15.3f} {imp_val:<15.3f} {change_pct:+.2f}%")
    
    print("=" * 80 + "\n")
    
    # Extract waiting times (weighted by demand)
    baseline_waiting = baseline_scores.get("waiting", {})
    improved_waiting = improved_scores.get("waiting", {})
    
    baseline_waits = []
    for stop_id, w in baseline_waiting.items():
        demand = baseline.get("stops_dict", {}).get(stop_id, {}).get("demand", 1)
        baseline_waits.append(w * demand)
    
    alns_waits = []
    for stop_id, w in improved_waiting.items():
        demand = improved.get("stops_dict", {}).get(stop_id, {}).get("demand", 1)
        alns_waits.append(w * demand)
    
    # Generate plot
    output_dir = Path(args.output_dir)
    output_path = output_dir / "baseline_vs_alns_wait_hist"
    
    plot_baseline_vs_alns(
        baseline_waits=baseline_waits,
        alns_waits=alns_waits,
        city_name=args.city,
        output_path=output_path
    )
    
    print(f"Plot saved to: {output_path}.png")
    if (output_path.parent / (output_path.name + ".html")).exists():
        print(f"Interactive plot: {output_path}.html")


if __name__ == "__main__":
    main()

