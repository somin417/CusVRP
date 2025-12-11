"""
Configuration management for VRP fairness experiments.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import os


class CityConfig(BaseModel):
    """City configuration with realistic bounding box and default DC."""
    name: str
    lat_min: float = Field(..., description="Minimum latitude")
    lat_max: float = Field(..., description="Maximum latitude")
    lon_min: float = Field(..., description="Minimum longitude")
    lon_max: float = Field(..., description="Maximum longitude")
    default_dc_lat: Optional[float] = Field(None, description="Default DC latitude")
    default_dc_lon: Optional[float] = Field(None, description="Default DC longitude")


class DCConfig(BaseModel):
    """Distribution center configuration."""
    id: str
    lat: float = Field(..., ge=-90, le=90, description="Latitude")
    lon: float = Field(..., ge=-180, le=180, description="Longitude")
    name: Optional[str] = None


class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    seed: int = Field(default=0, description="Random seed for reproducibility")
    n_stops: int = Field(default=60, gt=0, description="Number of stops")
    city: str = Field(default="daejeon", description="City name for bounding box")
    dcs: List[str] = Field(..., description="DC coordinates as 'lat,lon' strings")
    vehicles_per_dc: int = Field(default=3, gt=0, description="Number of vehicles per DC")
    
    # Fairness algorithm parameters
    eps: float = Field(default=0.10, ge=0.0, description="Cost budget tolerance (1+eps)")
    max_iters: int = Field(default=300, gt=0, description="Maximum local search iterations")
    lambda_balance: float = Field(default=0.1, ge=0.0, description="Balance penalty weight")
    
    # Data source
    stops_file: Optional[str] = Field(default=None, description="CSV file path for stops (optional)")
    approx_mode: bool = Field(default=False, description="Use haversine instead of iNavi")
    
    # Output settings
    output_dir: str = Field(default="outputs", description="Output directory")
    
    @field_validator("dcs")
    @classmethod
    def parse_dcs(cls, v: List[str]) -> List[DCConfig]:
        """Parse DC coordinate strings into DCConfig objects.
        
        Supports formats:
        - "lat,lon" -> DC1, DC2, ...
        - "lat,lon,name" -> uses provided name
        """
        dcs = []
        for i, dc_str in enumerate(v):
            try:
                parts = dc_str.strip().split(",")
                if len(parts) == 2:
                    # Format: lat,lon
                    lat, lon = float(parts[0]), float(parts[1])
                    dc_id = f"DC{i+1}"
                    name = dc_id
                elif len(parts) == 3:
                    # Format: lat,lon,name
                    lat, lon = float(parts[0]), float(parts[1])
                    dc_id = parts[2].strip()
                    name = dc_id
                else:
                    raise ValueError(f"Invalid DC format: {dc_str} (expected 'lat,lon' or 'lat,lon,name')")
                dcs.append(DCConfig(id=dc_id, lat=lat, lon=lon, name=name))
            except ValueError as e:
                raise ValueError(f"Invalid DC coordinates '{dc_str}': {e}")
        return dcs
    
    @property
    def dc_list(self) -> List[DCConfig]:
        """Get list of DC configurations."""
        return self.dcs  # Already parsed by validator
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True


# City configurations with realistic bounding boxes
CITY_CONFIGS = {
    "daejeon": CityConfig(
        name="Daejeon",
        lat_min=36.25,  # Realistic bounds covering central Daejeon
        lat_max=36.45,
        lon_min=127.30,
        lon_max=127.50,
        default_dc_lat=36.3500,  # Central Daejeon (Yuseong-gu area)
        default_dc_lon=127.3850
    ),
    "seoul": CityConfig(
        name="Seoul",
        lat_min=37.40,
        lat_max=37.70,
        lon_min=126.80,
        lon_max=127.20,
        default_dc_lat=37.5665,
        default_dc_lon=126.9780
    ),
    "busan": CityConfig(
        name="Busan",
        lat_min=35.00,
        lat_max=35.30,
        lon_min=129.00,
        lon_max=129.30,
        default_dc_lat=35.1796,
        default_dc_lon=129.0756
    ),
}

# Legacy bounding boxes for backward compatibility
CITY_BOUNDS = {
    "daejeon": (36.25, 36.45, 127.30, 127.50),
    "seoul": (37.40, 37.70, 126.80, 127.20),
    "busan": (35.00, 35.30, 129.00, 129.30),
    "default": (36.25, 36.45, 127.30, 127.50),  # Daejeon default
}


def get_city_config(city: str) -> CityConfig:
    """Get city configuration."""
    city_lower = city.lower()
    if city_lower in CITY_CONFIGS:
        return CITY_CONFIGS[city_lower]
    # Return Daejeon as default
    return CITY_CONFIGS["daejeon"]


def get_city_bounds(city: str) -> tuple[float, float, float, float]:
    """Get bounding box for a city (legacy function)."""
    config = get_city_config(city)
    return (config.lat_min, config.lat_max, config.lon_min, config.lon_max)

