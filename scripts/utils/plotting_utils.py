"""
Common plotting utilities for scripts.
Extracted from duplicate plotting code across multiple scripts.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None

logger = logging.getLogger(__name__)


def calc_bins(*series: List[float]) -> np.ndarray:
    """Calculate bins for histogram from multiple data series."""
    combined = []
    for s in series:
        combined.extend(s)
    if not combined:
        return np.array([0, 1])
    max_wait = max(combined)
    bin_count = min(60, max(20, len(combined) // 3))
    return np.linspace(0, max_wait * 1.05, bin_count)


def smooth_curve(x: np.ndarray, y: np.ndarray, num_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create smooth curve from discrete points using interpolation.
    Uses scipy.interpolate.UnivariateSpline if available, otherwise numpy interpolation.
    """
    if len(x) == 0 or len(y) == 0:
        return x, y
    
    if len(x) == 1:
        return x, y
    
    # Filter out zero counts at edges for better smoothing
    mask = np.ones(len(y), dtype=bool)
    if len(y) > 2:
        for i in range(1, len(y) - 1):
            if y[i] == 0 and y[i-1] == 0 and y[i+1] == 0:
                mask[i] = False
    
    x_filtered = x[mask]
    y_filtered = y[mask]
    
    if len(x_filtered) < 2:
        return x, y
    
    try:
        from scipy.interpolate import UnivariateSpline
        s = max(1, len(y_filtered) * 0.5)
        spline = UnivariateSpline(x_filtered, y_filtered, s=s, k=min(3, len(x_filtered) - 1))
        x_smooth = np.linspace(x[0], x[-1], num_points)
        y_smooth = spline(x_smooth)
        y_smooth = np.maximum(y_smooth, 0)
        return x_smooth, y_smooth
    except ImportError:
        try:
            from scipy.interpolate import interp1d
            kind = 'cubic' if len(x_filtered) >= 4 else 'linear'
            interp_func = interp1d(x_filtered, y_filtered, kind=kind, bounds_error=False, fill_value=0)
            x_smooth = np.linspace(x[0], x[-1], num_points)
            y_smooth = interp_func(x_smooth)
            y_smooth = np.maximum(y_smooth, 0)
            return x_smooth, y_smooth
        except (ImportError, ValueError):
            x_smooth = np.linspace(x[0], x[-1], num_points)
            y_smooth = np.interp(x_smooth, x_filtered, y_filtered)
            y_smooth = np.maximum(y_smooth, 0)
            return x_smooth, y_smooth


def plot_waiting_histogram(
    waiting_by_method: Dict[str, List[float]],
    output_path: Path,
    weights_by_method: Optional[Dict[str, List[float]]] = None,
    city_name: str = "daejeon",
    title_suffix: str = "Waiting Time Distribution"
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Plot waiting time histogram (raw waiting times, household-weighted frequency).
    
    Args:
        waiting_by_method: Dict mapping method name -> list of raw waiting times
        weights_by_method: Optional dict mapping method name -> list of household weights
        output_path: Output file path (without extension)
        city_name: City name for title
        title_suffix: Title suffix
    
    Returns:
        Tuple of (bins, counts_by_method)
    """
    if not waiting_by_method:
        logger.error("No data to plot!")
        return np.array([]), {}
    
    # Calculate bins
    all_waits = []
    for waits in waiting_by_method.values():
        all_waits.extend(waits)
    
    if not all_waits:
        return np.array([]), {}
    
    bins = calc_bins(*waiting_by_method.values())
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
    centers = bins[:-1] + bin_width / 2
    
    # Method colors
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
        "Variance-Improved": "#F58518",
        "MAD-Improved": "#54A24B",
    }
    
    # Calculate histogram counts (with weights if provided)
    counts_by_method = {}
    for method, waits in waiting_by_method.items():
        weights = weights_by_method.get(method) if weights_by_method else None
        counts, _ = np.histogram(waits, bins=bins, weights=weights)
        counts_by_method[method] = counts
    
    # Find global max for consistent y-scale
    global_max = 0
    for counts in counts_by_method.values():
        if len(counts) > 0:
            global_max = max(global_max, max(counts))
    global_cap = global_max * 1.05 if global_max > 0 else None
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot bars (side-by-side)
    methods = sorted(waiting_by_method.keys())
    num_methods = len(methods)
    offset_step = bin_width * 0.25
    
    for i, method in enumerate(methods):
        counts = counts_by_method[method]
        color = method_colors.get(method, plt.cm.tab10(i % 10))
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
    ax.set_xlabel('Waiting Time (seconds)', fontsize=12)
    ax.set_ylabel('Frequency (household-weighted)' if weights_by_method else 'Frequency', fontsize=12)
    ax.set_title(f'{title_suffix} ({city_name.title()})', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    
    if global_cap is not None:
        ax.set_ylim(top=global_cap)
    
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    # Save PNG
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path) + ".png", format="png", dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved waiting histogram: {output_path}.png")
    
    # Create interactive HTML with Plotly
    if PLOTLY_AVAILABLE:
        try:
            fig_plotly = go.Figure()
            
            for method, waits in waiting_by_method.items():
                color = method_colors.get(method, None)
                fig_plotly.add_trace(go.Histogram(
                    x=waits,
                    name=method,
                    opacity=0.75,
                    marker_color=color,
                    nbinsx=min(60, max(20, len(waits) // 3))
                ))
            
            fig_plotly.update_layout(
                title=f'{title_suffix} ({city_name.title()})',
                xaxis_title='Waiting Time (seconds)',
                yaxis_title='Frequency (household-weighted)' if weights_by_method else 'Frequency',
                barmode='overlay',
                hovermode='x unified',
                width=1200,
                height=600
            )
            
            # HTML generation removed - only PNG plots are generated
            # Maps are the only place where HTML is generated
            # fig_plotly.write_html(str(output_path) + ".html", include_plotlyjs="cdn")
            # logger.info(f"Saved interactive histogram: {output_path}.html")
        except Exception as e:
            logger.warning(f"Failed to create interactive plot: {e}")
    else:
        logger.info("Plotly not available, skipping interactive HTML generation")
    
    return bins, counts_by_method


def plot_weighted_waiting_histogram(
    weighted_by_method: Dict[str, List[float]],
    output_path: Path,
    city_name: str = "daejeon"
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Plot weighted waiting time histogram (weighted waiting times, frequency count).
    
    Args:
        weighted_by_method: Dict mapping method name -> list of weighted waiting times
        output_path: Output file path (without extension)
        city_name: City name for title
    
    Returns:
        Tuple of (bins, counts_by_method)
    """
    if not weighted_by_method:
        logger.error("No data to plot!")
        return np.array([]), {}
    
    # Collect all weighted waiting times to determine bins
    all_weighted_waits = []
    for waits in weighted_by_method.values():
        all_weighted_waits.extend(waits)
    
    if not all_weighted_waits:
        return np.array([]), {}
    
    max_wait = max(all_weighted_waits)
    min_wait = min(all_weighted_waits)
    bin_count = min(60, max(20, len(all_weighted_waits) // 3))
    bins = np.linspace(min_wait, max_wait * 1.05, bin_count)
    
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 1.0
    centers = bins[:-1] + bin_width / 2
    
    # Method colors
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
    
    # Calculate histogram counts
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
    
    # Plot bars (side-by-side)
    methods = sorted(weighted_by_method.keys())
    num_methods = len(methods)
    offset_step = bin_width * 0.25
    
    for i, method in enumerate(methods):
        counts = counts_by_method[method]
        color = method_colors.get(method, plt.cm.tab10(i % 10))
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
    ax.set_title(f'Weighted Waiting Time Distribution ({city_name.title()})', fontsize=14, fontweight='bold')
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
    
    # Create interactive HTML with Plotly
    if PLOTLY_AVAILABLE:
        try:
            fig_plotly = go.Figure()
            
            for method, weighted_waits in weighted_by_method.items():
                color = method_colors.get(method, None)
                fig_plotly.add_trace(go.Histogram(
                    x=weighted_waits,
                    name=method,
                    opacity=0.6,
                    marker_color=color,
                    nbinsx=bin_count
                ))
            
            fig_plotly.update_layout(
                barmode='overlay',
                title=f'Weighted Waiting Time Distribution ({city_name.title()})',
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
            # fig_plotly.write_html(str(output_path) + ".html", include_plotlyjs="cdn")
            # logger.info(f"Saved interactive histogram: {output_path}.html")
        except Exception as e:
            logger.warning(f"Failed to create interactive plot: {e}")
    else:
        logger.info("Plotly not available, skipping interactive HTML generation")
    
    return bins, counts_by_method

