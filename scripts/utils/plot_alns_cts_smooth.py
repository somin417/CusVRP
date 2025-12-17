#!/usr/bin/env python3
"""
Generate smooth curve plots for waiting time and weighted waiting time.
Compares only ALNS (Improved) and CTS solutions.

Usage:
    python scripts/utils/plot_alns_cts_smooth.py --baseline outputs/solutions/baseline.json --improved outputs/solutions/ALNS_MAD.json --cts outputs/solutions/cts_solution.json
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.utils import load_json, extract_waiting_times_with_weights, calculate_weighted_waiting_times
from scripts.utils.plotting_utils import smooth_curve, calc_bins
from src.vrp_fairness.objectives import compute_waiting_times
from src.vrp_fairness.osrm_provider import create_osrm_providers
from src.vrp_fairness.inavi import iNaviCache

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def plot_smooth_curves(
    alns_waits: List[float],
    cts_waits: List[float],
    alns_weighted: List[float],
    cts_weighted: List[float],
    alns_weights: List[float],
    cts_weights: List[float],
    output_path: Path,
    city_name: str = "daejeon"
) -> None:
    """
    Create smooth curve plots for waiting time and weighted waiting time.
    
    Args:
        alns_waits: List of raw waiting times for ALNS
        cts_waits: List of raw waiting times for CTS
        alns_weighted: List of weighted waiting times for ALNS
        cts_weighted: List of weighted waiting times for CTS
        alns_weights: List of household weights for ALNS
        cts_weights: List of household weights for CTS
        output_path: Output file path (without extension)
        city_name: City name for title
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Colors
    alns_color = "#54A24B"  # Green for ALNS
    cts_color = "#E45756"   # Red for CTS
    
    # ===== Plot 1: Waiting Time (Raw) =====
    all_waits = alns_waits + cts_waits
    if all_waits:
        bins = calc_bins(alns_waits, cts_waits)
        bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
        centers = bins[:-1] + bin_width / 2
        
        # Calculate histogram counts (with weights for household-weighted frequency)
        alns_counts, _ = np.histogram(alns_waits, bins=bins, weights=alns_weights)
        cts_counts, _ = np.histogram(cts_waits, bins=bins, weights=cts_weights)
        
        # Plot bars (side-by-side)
        offset = bin_width * 0.25
        bar_width = bin_width * 0.4
        ax1.bar(centers - offset, alns_counts, width=bar_width, 
               color=alns_color, alpha=0.6, label='ALNS', 
               edgecolor='white', linewidth=0.7)
        ax1.bar(centers + offset, cts_counts, width=bar_width, 
               color=cts_color, alpha=0.6, label='CTS', 
               edgecolor='white', linewidth=0.7)
        
        # Create smooth curves
        alns_x_smooth, alns_y_smooth = smooth_curve(centers, alns_counts)
        cts_x_smooth, cts_y_smooth = smooth_curve(centers, cts_counts)
        
        # Plot smooth curves on top
        ax1.plot(alns_x_smooth, alns_y_smooth, color=alns_color, linewidth=2.5, 
                alpha=0.9, zorder=10)
        ax1.plot(cts_x_smooth, cts_y_smooth, color=cts_color, linewidth=2.5, 
                alpha=0.9, zorder=10)
        
        ax1.set_xlabel('Waiting Time (seconds)', fontsize=12)
        ax1.set_ylabel('Frequency (household-weighted)', fontsize=12)
        ax1.set_title(f'Waiting Time Distribution ({city_name.title()})', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        alns_mean = np.mean(alns_waits)
        alns_max = np.max(alns_waits)
        cts_mean = np.mean(cts_waits)
        cts_max = np.max(cts_waits)
        
        stats_text = (
            f'ALNS: Mean={alns_mean:.1f}s, Max={alns_max:.1f}s\n'
            f'CTS: Mean={cts_mean:.1f}s, Max={cts_max:.1f}s'
        )
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                ha='left', va='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    # ===== Plot 2: Weighted Waiting Time =====
    all_weighted = alns_weighted + cts_weighted
    if all_weighted:
        max_wait = max(all_weighted)
        min_wait = min(all_weighted)
        bin_count = min(60, max(20, len(all_weighted) // 3))
        bins = np.linspace(min_wait, max_wait * 1.05, bin_count)
        bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
        centers = bins[:-1] + bin_width / 2
        
        # Calculate histogram counts
        alns_counts, _ = np.histogram(alns_weighted, bins=bins)
        cts_counts, _ = np.histogram(cts_weighted, bins=bins)
        
        # Plot bars (side-by-side)
        offset = bin_width * 0.25
        bar_width = bin_width * 0.4
        ax2.bar(centers - offset, alns_counts, width=bar_width, 
               color=alns_color, alpha=0.6, label='ALNS', 
               edgecolor='white', linewidth=0.7)
        ax2.bar(centers + offset, cts_counts, width=bar_width, 
               color=cts_color, alpha=0.6, label='CTS', 
               edgecolor='white', linewidth=0.7)
        
        # Create smooth curves
        alns_x_smooth, alns_y_smooth = smooth_curve(centers, alns_counts)
        cts_x_smooth, cts_y_smooth = smooth_curve(centers, cts_counts)
        
        # Plot smooth curves on top
        ax2.plot(alns_x_smooth, alns_y_smooth, color=alns_color, linewidth=2.5, 
                alpha=0.9, zorder=10)
        ax2.plot(cts_x_smooth, cts_y_smooth, color=cts_color, linewidth=2.5, 
                alpha=0.9, zorder=10)
        
        ax2.set_xlabel('Weighted Waiting Time (seconds × households)', fontsize=12)
        ax2.set_ylabel('Frequency (count)', fontsize=12)
        ax2.set_title(f'Weighted Waiting Time Distribution ({city_name.title()})', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add statistics text
        alns_mean = np.mean(alns_weighted)
        alns_max = np.max(alns_weighted)
        cts_mean = np.mean(cts_weighted)
        cts_max = np.max(cts_weighted)
        
        stats_text = (
            f'ALNS: Mean={alns_mean:.1f}, Max={alns_max:.1f}\n'
            f'CTS: Mean={cts_mean:.1f}, Max={cts_max:.1f}'
        )
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                ha='left', va='top', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    plt.tight_layout()
    
    # Save PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + ".png", format="png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved smooth curve plots: {output_path}.png")


def main():
    parser = argparse.ArgumentParser(
        description="Generate smooth curve plots for ALNS and CTS"
    )
    parser.add_argument("--baseline", type=str, required=True,
                       help="Path to baseline.json (required for depots/stops)")
    parser.add_argument("--improved", type=str, required=True,
                       help="Path to ALNS_MAD.json (ALNS solution)")
    parser.add_argument("--cts", type=str, required=True,
                       help="Path to cts_solution.json")
    parser.add_argument("--output", type=str, default="outputs/alns_cts_smooth",
                       help="Output file path (without extension, default: outputs/alns_cts_smooth)")
    parser.add_argument("--city", type=str, default="daejeon",
                       help="City name for plots (default: daejeon)")
    
    args = parser.parse_args()
    
    # Load solutions
    baseline_path = Path(args.baseline)
    improved_path = Path(args.improved)
    cts_path = Path(args.cts)
    
    if not baseline_path.exists():
        logger.error(f"Baseline file not found: {baseline_path}")
        sys.exit(1)
    
    if not improved_path.exists():
        logger.error(f"Improved file not found: {improved_path}")
        sys.exit(1)
    
    if not cts_path.exists():
        logger.error(f"CTS file not found: {cts_path}")
        sys.exit(1)
    
    logger.info(f"Loading baseline from: {baseline_path}")
    baseline = load_json(baseline_path)
    
    logger.info(f"Loading ALNS (improved) from: {improved_path}")
    improved = load_json(improved_path)
    
    logger.info(f"Loading CTS from: {cts_path}")
    cts = load_json(cts_path)
    
    # Extract depots and stops_dict from baseline
    depots = baseline.get("depots", [])
    stops_by_id = baseline.get("stops_dict", {})
    
    if not depots:
        logger.error("No depots found in baseline solution!")
        sys.exit(1)
    
    if not stops_by_id:
        logger.error("No stops_dict found in baseline solution!")
        sys.exit(1)
    
    logger.info(f"Found {len(depots)} depots and {len(stops_by_id)} stops")
    
    # Create OSRM providers
    logger.info("Creating OSRM providers...")
    cache = iNaviCache(approx_mode=False)
    time_provider, distance_provider = create_osrm_providers(depots, stops_by_id, cache)
    
    # Compute waiting times for ALNS
    logger.info("Computing waiting times for ALNS...")
    alns_waiting = compute_waiting_times(improved, stops_by_id, time_provider)
    alns_waits, alns_weights = extract_waiting_times_with_weights(alns_waiting, stops_by_id)
    alns_weighted = calculate_weighted_waiting_times(alns_waiting, stops_by_id)
    logger.info(f"  ALNS: {len(alns_waits)} stops")
    
    # Compute waiting times for CTS
    logger.info("Computing waiting times for CTS...")
    cts_waiting = compute_waiting_times(cts, stops_by_id, time_provider)
    cts_waits, cts_weights = extract_waiting_times_with_weights(cts_waiting, stops_by_id)
    cts_weighted = calculate_weighted_waiting_times(cts_waiting, stops_by_id)
    logger.info(f"  CTS: {len(cts_waits)} stops")
    
    # Generate plots
    output_path = Path(args.output)
    plot_smooth_curves(
        alns_waits=alns_waits,
        cts_waits=cts_waits,
        alns_weighted=alns_weighted,
        cts_weighted=cts_weighted,
        alns_weights=alns_weights,
        cts_weights=cts_weights,
        output_path=output_path,
        city_name=args.city
    )
    
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"  Smooth curve plots: {output_path}.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
