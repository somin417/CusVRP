"""
Data loading and generation for stops.
"""

import random
import csv
import sqlite3
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from .config import ExperimentConfig, get_city_bounds, get_city_config, CityConfig, DCConfig


@dataclass
class Stop:
    """Stop data structure."""
    id: str
    lat: float
    lon: float
    service_time: int = 300  # Default 5 minutes in seconds
    demand: int = 1  # Default demand
    meta: Dict = field(default_factory=dict)  # Optional metadata
    
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


def load_stops_from_gpkg(
    gpkg_path: str,
    layer: str = "yuseong_housing_2__point",
    n: Optional[int] = None,
    seed: int = 0,
    housing_type: Optional[str] = None,
    demand_field: Optional[str] = None,
    service_time_s: int = 300,
) -> List[Stop]:
    """
    Load stops from GeoPackage file using sqlite3 + pandas.
    
    Args:
        gpkg_path: Path to GeoPackage file
        layer: Table/layer name (default: "yuseong_housing_2__point")
        n: Number of rows to sample (None = all)
        seed: Random seed for sampling
        housing_type: Filter by A9 column ("공동주택" or "단독주택")
        demand_field: Column to use for demand (e.g., "A26" or None for demand=1)
        service_time_s: Service time per stop in seconds
    
    Returns:
        List of Stop objects
    """
    import pandas as pd
    
    # Connect to GeoPackage (it's a SQLite database)
    conn = sqlite3.connect(gpkg_path)
    
    # Build query
    columns = ["fid", "latitude", "longitude", "A4", "A9", "A26"]
    query = f"SELECT {', '.join(columns)} FROM {layer}"
    
    # Add housing type filter
    if housing_type:
        query += f" WHERE A9 = '{housing_type}'"
    
    # Execute query and load into pandas
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        raise ValueError(f"No rows found in {layer}" + (f" with A9='{housing_type}'" if housing_type else ""))
    
    # Sample if requested
    if n is not None:
        if n > len(df):
            raise ValueError(f"Requested {n} samples but only {len(df)} rows available")
        df = df.sample(n=n, random_state=seed)
    
    # Convert to Stop objects
    stops = []
    for _, row in df.iterrows():
        fid = int(row['fid'])
        lat = float(row['latitude'])
        lon = float(row['longitude'])
        
        # Determine demand
        if demand_field == "A26" and pd.notna(row['A26']):
            demand = int(round(float(row['A26'])))
        else:
            demand = 1
        
        # Store metadata
        meta = {
            "address": str(row['A4']) if pd.notna(row['A4']) else "",
            "housing_type": str(row['A9']) if pd.notna(row['A9']) else "",
            "A26": float(row['A26']) if pd.notna(row['A26']) else None
        }
        
        stops.append(Stop(
            id=str(fid),
            lat=lat,
            lon=lon,
            service_time=service_time_s,
            demand=demand,
            meta=meta
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

