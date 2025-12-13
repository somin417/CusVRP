"""
Common utilities for scripts.
Extracted from duplicate code across multiple scripts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def extract_waiting_times_with_weights(
    waiting: Dict[str, float],
    stops_by_id: Dict[str, Dict[str, Any]]
) -> Tuple[List[float], List[float]]:
    """
    Extract raw waiting times and household weights.
    
    Args:
        waiting: Dict mapping stop_id -> waiting_time_seconds
        stops_by_id: Dict mapping stop_id -> stop data (with households/demand)
    
    Returns:
        Tuple of (waiting_times, weights) lists
    """
    vals, wts = [], []
    for stop_id, w in waiting.items():
        wt = stops_by_id.get(stop_id, {}).get("households", stops_by_id.get(stop_id, {}).get("demand", 1))
        if wt in (None, 0):
            wt = 1
        vals.append(w)  # Raw waiting time
        wts.append(wt)  # Household weight
    return vals, wts


def calculate_weighted_waiting_times(
    waiting: Dict[str, float],
    stops_by_id: Dict[str, Dict[str, Any]]
) -> List[float]:
    """
    Calculate weighted waiting times (waiting_time * households).
    
    Args:
        waiting: Dict mapping stop_id -> waiting_time_seconds
        stops_by_id: Dict mapping stop_id -> stop data (with households/demand)
    
    Returns:
        List of weighted waiting times
    """
    weighted = []
    for stop_id, w in waiting.items():
        wt = stops_by_id.get(stop_id, {}).get("households", stops_by_id.get(stop_id, {}).get("demand", 1))
        if wt in (None, 0):
            wt = 1
        weighted.append(w * wt)
    return weighted

