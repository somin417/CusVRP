#!/usr/bin/env python3
"""
Generate weighted graph (y: frequency, x: weighted waiting time) from results
of three test scripts:
- compare_cts_vs_alns.py
- compare_waiting_and_scores.py
- compare_Z3_variance_vs_MAD.py

Each script saves CSV files with columns: method, waiting_time_seconds, households
This script reads those CSVs, calculates weighted waiting time = waiting_time_seconds * households,
and plots a histogram showing frequency distribution of weighted waiting times.
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_wait_values_csv(csv_path: Path) -> List[Tuple[str, float, float]]:
    """
    Load wait values CSV file.
    
    Args:
        csv_path: Path to CSV file with columns: method, waiting_time_seconds, households
    
    Returns:
        List of tuples: (method, waiting_time_seconds, households)
    """
    if not csv_path.exists():
        logger.warning(f"CSV file not found: {csv_path}")
        return []
    
    data = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                method = row.get('method', '').strip()
                waiting_time = float(row.get('waiting_time_seconds', 0))
                households = float(row.get('households', 1))
                if households <= 0:
                    households = 1
                data.append((method, waiting_time, households))
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid row in {csv_path}: {row}, error: {e}")
    
    logger.info(f"Loaded {len(data)} rows from {csv_path}")
    return data


def calculate_weighted_waiting_times(data: List[Tuple[str, float, float]]) -> Dict[str, List[float]]:
    """
    Calculate weighted waiting times for each method.
    
    Args:
        data: List of (method, waiting_time_seconds, households) tuples
    
    Returns:
        Dictionary mapping method -> list of weighted waiting times
    """
    weighted_by_method = {}
    
    for method, waiting_time, households in data:
        weighted_wait = waiting_time * households
        if method not in weighted_by_method:
            weighted_by_method[method] = []
        weighted_by_method[method].append(weighted_wait)
    
    # Log summary
    for method, weighted_waits in weighted_by_method.items():
        if weighted_waits:
            logger.info(f"{method}: {len(weighted_waits)} stops, "
                       f"weighted wait range: [{min(weighted_waits):.1f}, {max(weighted_waits):.1f}], "
                       f"mean: {np.mean(weighted_waits):.1f}")
    
    return weighted_by_method


def plot_weighted_waiting_histogram(
    weighted_by_method: Dict[str, List[float]],
    output_path: Path,
    city: str = "daejeon",
    bins: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Plot histogram of weighted waiting times.
    
    Args:
        weighted_by_method: Dictionary mapping method -> list of weighted waiting times
        output_path: Output file path (without extension)
        city: City name for title
        bins: Optional pre-computed bins (if None, will compute from data)
    
    Returns:
        Tuple of (bins, counts_by_method) where counts_by_method maps method -> histogram counts
    """
    if not weighted_by_method:
        logger.error("No data to plot!")
        return np.array([]), {}
    
    # Collect all weighted waiting times to determine bins
    all_weighted_waits = []
    for waits in weighted_by_method.values():
        all_weighted_waits.extend(waits)
    
    if not all_weighted_waits:
        logger.error("No weighted waiting times to plot!")
        return np.array([]), {}
    
    # Calculate bins if not provided
    if bins is None:
        max_wait = max(all_weighted_waits)
        min_wait = min(all_weighted_waits)
        bin_count = min(60, max(20, len(all_weighted_waits) // 3))
        bins = np.linspace(min_wait, max_wait * 1.05, bin_count)
    
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
    centers = bins[:-1] + bin_width / 2
    
    # Define colors for different methods
    method_colors = {
        "baseline": "#4C78A8",
        "Baseline": "#4C78A8",
        "local": "#54A24B",
        "Local": "#54A24B",
        "ALSM (MAD)": "#F58518",
        "ALNS (fixed)": "#54A24B",
        "CTS": "#E45756",
        "Variance": "#F58518",
        "MAD": "#54A24B",
    }
    
    # Calculate histogram counts for each method
    counts_by_method = {}
    for method, weighted_waits in weighted_by_method.items():
        counts, _ = np.histogram(weighted_waits, bins=bins)
        counts_by_method[method] = counts
    
    # Find global max for consistent y-scale
    global_max = 0
    for counts in counts_by_method.values():
        if len(counts) > 0:
            global_max = max(global_max, max(counts))
    global_cap = global_max * 1.05 if global_max > 0 else None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot bars for each method (side-by-side)
    offset_step = bin_width * 0.25
    methods = sorted(weighted_by_method.keys())
    num_methods = len(methods)
    
    for i, method in enumerate(methods):
        counts = counts_by_method[method]
        color = method_colors.get(method, plt.cm.tab10(i % 10))
        
        # Calculate offset for side-by-side bars
        offset = (i - (num_methods - 1) / 2) * offset_step
        bar_width = bin_width * 0.8 / num_methods if num_methods > 1 else bin_width * 0.8
        
        ax.bar(
            centers + offset,
            counts,
            width=bar_width,
            alpha=0.75,
            color=color,
            label=method,
            edgecolor='white',
            linewidth=0.7
        )
    
    # Formatting
    ax.set_xlabel('Weighted Waiting Time (seconds × households)', fontsize=12)
    ax.set_ylabel('Frequency (count)', fontsize=12)
    ax.set_title(f'Weighted Waiting Time Distribution ({city.title()})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    if global_cap is not None:
        ax.set_ylim(top=global_cap)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + ".png", format="png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved weighted waiting histogram: {output_path}.png")
    
    return bins, counts_by_method


def plot_weighted_waiting_interactive(
    weighted_by_method: Dict[str, List[float]],
    output_path: Path,
    city: str = "daejeon"
) -> None:
    """Create interactive Plotly histogram."""
    if not PLOTLY_AVAILABLE:
        logger.info("Plotly not available, skipping interactive histogram")
        return
    
    if not weighted_by_method:
        return
    
    fig = go.Figure()
    
    # Collect all data for bin calculation
    all_weighted_waits = []
    for waits in weighted_by_method.values():
        all_weighted_waits.extend(waits)
    
    if not all_weighted_waits:
        return
    
    max_wait = max(all_weighted_waits)
    min_wait = min(all_weighted_waits)
    bin_count = min(60, max(20, len(all_weighted_waits) // 3))
    
    method_colors = {
        "baseline": "#4C78A8",
        "Baseline": "#4C78A8",
        "local": "#54A24B",
        "Local": "#54A24B",
        "ALSM (MAD)": "#F58518",
        "ALNS (fixed)": "#54A24B",
        "CTS": "#E45756",
        "Variance": "#F58518",
        "MAD": "#54A24B",
    }
    
    for method, weighted_waits in weighted_by_method.items():
        color = method_colors.get(method, None)
        fig.add_trace(go.Histogram(
            x=weighted_waits,
            name=method,
            opacity=0.6,
            marker_color=color,
            nbinsx=bin_count
        ))
    
    fig.update_layout(
        barmode='overlay',
        title=f'Weighted Waiting Time Distribution ({city.title()})',
        xaxis_title='Weighted Waiting Time (seconds × households)',
        yaxis_title='Frequency (count)',
        legend_title='Method',
        template='plotly_white',
        hovermode='x unified',
        width=1200,
        height=600
    )
    
    # HTML generation removed - only PNG plots are generated
    # Maps are the only place where HTML is generated
    # fig.write_html(str(output_path) + ".html", include_plotlyjs="cdn")
    # logger.info(f"Saved interactive histogram: {output_path}.html")


def main():
    parser = argparse.ArgumentParser(
        description="Generate weighted waiting time graph from three test scripts' results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory where CSV files are located and where plots will be saved"
    )
    parser.add_argument(
        "--city",
        type=str,
        default="daejeon",
        help="City name for plot title"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="weighted_waiting_graph",
        help="Output file name (without extension)"
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create organized subdirectories
    data_dir = output_dir / "data"
    plots_dir = output_dir / "plots"
    data_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Define CSV file paths from the three scripts (in data directory)
    csv_files = {
        "compare_waiting_and_scores": data_dir / "baseline_local_alns_mad_wait_values.csv",
        "compare_cts_vs_alns": data_dir / "cts_vs_alns_wait_values.csv",
        "compare_Z3_variance_vs_MAD": data_dir / "baseline_alns_variance_vs_mad_wait_values.csv",
    }
    
    # Load data from all CSV files
    all_data = []
    for script_name, csv_path in csv_files.items():
        logger.info(f"Loading data from {script_name}: {csv_path}")
        data = load_wait_values_csv(csv_path)
        if data:
            all_data.extend(data)
            logger.info(f"  Loaded {len(data)} rows")
        else:
            logger.warning(f"  No data found in {csv_path}")
    
    if not all_data:
        logger.error("No data loaded from any CSV file!")
        logger.error("Please run the three test scripts first:")
        logger.error("  - compare_waiting_and_scores.py")
        logger.error("  - compare_cts_vs_alns.py")
        logger.error("  - compare_Z3_variance_vs_MAD.py")
        sys.exit(1)
    
    logger.info(f"Total rows loaded: {len(all_data)}")
    
    # Calculate weighted waiting times
    weighted_by_method = calculate_weighted_waiting_times(all_data)
    
    if not weighted_by_method:
        logger.error("No weighted waiting times calculated!")
        sys.exit(1)
    
    # Plot histogram
    output_path = plots_dir / args.output_name
    bins, counts_by_method = plot_weighted_waiting_histogram(
        weighted_by_method,
        output_path,
        city=args.city
    )
    
    # Create interactive plot
    plot_weighted_waiting_interactive(
        weighted_by_method,
        output_path,
        city=args.city
    )
    
    # Save histogram data to CSV for later use
    hist_data_path = data_dir / f"{args.output_name}_hist_data.csv"
    with open(hist_data_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ["bin_left", "bin_right", "bin_center"] + list(weighted_by_method.keys())
        writer.writerow(header)
        
        # Data rows
        for i in range(len(bins) - 1):
            left = bins[i]
            right = bins[i + 1]
            center = (left + right) / 2
            row = [left, right, center]
            for method in weighted_by_method.keys():
                row.append(counts_by_method.get(method, np.array([]))[i] if i < len(counts_by_method.get(method, [])) else 0)
            writer.writerow(row)
    
    logger.info(f"Saved histogram data: {hist_data_path}")
    
    # Save weighted waiting time values to CSV
    weighted_values_path = data_dir / f"{args.output_name}_values.csv"
    with open(weighted_values_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "weighted_waiting_time"])
        for method, weighted_waits in weighted_by_method.items():
            for weighted_wait in weighted_waits:
                writer.writerow([method, weighted_wait])
    
    logger.info(f"Saved weighted waiting time values: {weighted_values_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Methods found: {', '.join(sorted(weighted_by_method.keys()))}")
    logger.info(f"Total stops: {sum(len(waits) for waits in weighted_by_method.values())}")
    logger.info(f"Output files:")
    logger.info(f"  - {output_path}.png")
    # HTML generation removed - only PNG plots are generated
    # Maps are the only place where HTML is generated
    logger.info(f"  - {hist_data_path}")
    logger.info(f"  - {weighted_values_path}")


if __name__ == "__main__":
    main()
