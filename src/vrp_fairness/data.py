"""
Data loading and generation for stops.
"""

import random
import csv
from typing import List, Dict, Optional
from dataclasses import dataclass

from .config import ExperimentConfig, get_city_bounds, get_city_config, CityConfig, DCConfig


@dataclass
class Stop:
    """Stop data structure."""
    id: str
    lat: float
    lon: float
    service_time: int = 300  # Default 5 minutes in seconds
    demand: int = 1  # Default demand
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "service_time": self.service_time,
            "demand": self.demand
        }


def generate_random_stops(
    n: int,
    city: str,
    seed: int = 0,
    service_time: int = 300
) -> List[Stop]:
    """
    Generate random stops within a city bounding box.
    
    Args:
        n: Number of stops to generate
        city: City name for bounding box
        seed: Random seed
        service_time: Service time per stop in seconds
    
    Returns:
        List of Stop objects
    """
    random.seed(seed)
    lat_min, lat_max, lon_min, lon_max = get_city_bounds(city)
    
    stops = []
    for i in range(1, n + 1):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        stops.append(Stop(
            id=f"Stop{i}",
            lat=lat,
            lon=lon,
            service_time=service_time,
            demand=1
        ))
    
    return stops


def generate_random_stops_in_city(
    city_config: CityConfig,
    n_stops: int,
    seed: int = 0,
    service_time: int = 300
) -> List[Stop]:
    """
    Generate random stops within a city configuration's bounding box.
    
    Args:
        city_config: CityConfig object with bounding box
        n_stops: Number of stops to generate
        seed: Random seed
        service_time: Service time per stop in seconds
    
    Returns:
        List of Stop objects
    """
    random.seed(seed)
    
    stops = []
    for i in range(1, n_stops + 1):
        lat = random.uniform(city_config.lat_min, city_config.lat_max)
        lon = random.uniform(city_config.lon_min, city_config.lon_max)
        stops.append(Stop(
            id=f"Stop{i}",
            lat=lat,
            lon=lon,
            service_time=service_time,
            demand=1
        ))
    
    return stops


def load_stops_from_csv(filepath: str) -> List[Stop]:
    """
    Load stops from CSV file.
    
    Expected CSV format:
    id,name,lat,lon,service_time,demand
    Stop1,Location1,37.5665,126.9780,300,1
    ...
    
    Or simplified format:
    id,lat,lon,service_time,demand
    Stop1,37.5665,126.9780,300,1
    ...
    
    Args:
        filepath: Path to CSV file
    
    Returns:
        List of Stop objects
    """
    stops = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            stops.append(Stop(
                id=row['id'],
                lat=float(row['lat']),
                lon=float(row['lon']),
                service_time=int(row.get('service_time', 300)),
                demand=int(row.get('demand', 1))
            ))
    return stops


def assign_stops_to_dcs(
    stops: List[Stop],
    dcs: List[DCConfig],
    distance_matrix_func: callable
) -> Dict[str, List[Stop]]:
    """
    Assign each stop to the nearest DC based on road distance.
    
    Args:
        stops: List of stops
        dcs: List of DC configurations
        distance_matrix_func: Function that takes (lat1, lon1, lat2, lon2) and returns distance
    
    Returns:
        Dictionary mapping DC ID to list of stops
    """
    assignments: Dict[str, List[Stop]] = {dc.id: [] for dc in dcs}
    
    for stop in stops:
        min_dist = float('inf')
        nearest_dc = dcs[0]
        
        for dc in dcs:
            dist = distance_matrix_func(dc.lat, dc.lon, stop.lat, stop.lon)
            if dist < min_dist:
                min_dist = dist
                nearest_dc = dc
        
        assignments[nearest_dc.id].append(stop)
    
    return assignments

