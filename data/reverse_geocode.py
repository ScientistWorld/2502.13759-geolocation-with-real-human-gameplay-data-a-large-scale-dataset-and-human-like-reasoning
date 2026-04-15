"""
Reverse geocoding utility for geolocation evaluation.

Provides city-level reverse geocoding from GPS coordinates.
Uses a local lookup table for known locations to avoid API calls.
"""

import math
import os
import csv
from typing import Dict, List, Optional, Tuple


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points (in km)."""
    R = 6371.0
    lat1_rad, lat2_rad = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Predefined city database for the GeoCLIP dataset locations
# These are the major cities/areas in the dataset countries
CITY_DATABASE = [
    # Kenya
    {"name": "Nairobi", "country": "Kenya", "continent": "Africa", "lat": -1.2921, "lon": 36.8219},
    {"name": "Mombasa", "country": "Kenya", "continent": "Africa", "lat": -4.0435, "lon": 39.6682},
    {"name": "Kisumu", "country": "Kenya", "continent": "Africa", "lat": -0.1022, "lon": 34.7617},
    {"name": "Nakuru", "country": "Kenya", "continent": "Africa", "lat": -0.3031, "lon": 36.0800},
    {"name": "Eldoret", "country": "Kenya", "continent": "Africa", "lat": 0.5143, "lon": 35.2698},
    {"name": "Malindi", "country": "Kenya", "continent": "Africa", "lat": -3.2138, "lon": 40.1169},
    {"name": "Lamu", "country": "Kenya", "continent": "Africa", "lat": -2.2686, "lon": 40.9020},
    {"name": "Diani Beach", "country": "Kenya", "continent": "Africa", "lat": -4.3124, "lon": 39.5835},
    {"name": "Watamu", "country": "Kenya", "continent": "Africa", "lat": -3.3452, "lon": 40.0303},
    {"name": "Tsavo", "country": "Kenya", "continent": "Africa", "lat": -2.9950, "lon": 38.5000},
    {"name": "Mombasa Coast", "country": "Kenya", "continent": "Africa", "lat": -4.0500, "lon": 39.6667},
    {"name": "Kilifi", "country": "Kenya", "continent": "Africa", "lat": -3.6305, "lon": 39.8499},
    {"name": "Kwale", "country": "Kenya", "continent": "Africa", "lat": -4.1783, "lon": 39.4520},
    {"name": "Kwale Coast", "country": "Kenya", "continent": "Africa", "lat": -4.2000, "lon": 39.4500},
    {"name": "Taita Taveta", "country": "Kenya", "continent": "Africa", "lat": -3.4000, "lon": 38.5000},
    {"name": "Voi", "country": "Kenya", "continent": "Africa", "lat": -3.3960, "lon": 38.5561},

    # Ecuador
    {"name": "Quito", "country": "Ecuador", "continent": "South America", "lat": -0.1807, "lon": -78.4678},
    {"name": "Guayaquil", "country": "Ecuador", "continent": "South America", "lat": -2.1709, "lon": -79.9224},
    {"name": "Cuenca", "country": "Ecuador", "continent": "South America", "lat": -2.9005, "lon": -79.0059},
    {"name": "Galapagos", "country": "Ecuador", "continent": "South America", "lat": -0.9538, "lon": -90.9656},
    {"name": "Manta", "country": "Ecuador", "continent": "South America", "lat": -0.9671, "lon": -80.7089},
    {"name": "Santo Domingo", "country": "Ecuador", "continent": "South America", "lat": -0.2389, "lon": -79.1780},
    {"name": "Ambato", "country": "Ecuador", "continent": "South America", "lat": -1.2408, "lon": -78.6270},
    {"name": "Riobamba", "country": "Ecuador", "continent": "South America", "lat": -1.6708, "lon": -78.6471},
    {"name": "Loja", "country": "Ecuador", "continent": "South America", "lat": -3.9931, "lon": -79.2011},
    {"name": "Ibarra", "country": "Ecuador", "continent": "South America", "lat": 0.3395, "lon": -78.1223},
    {"name": "Latacunga", "country": "Ecuador", "continent": "South America", "lat": -0.9303, "lon": -78.6152},
    {"name": "Tulcan", "country": "Ecuador", "continent": "South America", "lat": 0.8119, "lon": -77.7270},
    {"name": "Machala", "country": "Ecuador", "continent": "South America", "lat": -3.2586, "lon": -79.9555},
    {"name": "Esmeraldas", "country": "Ecuador", "continent": "South America", "lat": 0.9592, "lon": -79.6540},
    {"name": "Salinas", "country": "Ecuador", "continent": "South America", "lat": -2.2262, "lon": -80.9581},
    {"name": "Playas", "country": "Ecuador", "continent": "South America", "lat": -2.6561, "lon": -80.3926},

    # Chile
    {"name": "Santiago", "country": "Chile", "continent": "South America", "lat": -33.4489, "lon": -70.6693},
    {"name": "Valparaiso", "country": "Chile", "continent": "South America", "lat": -33.0458, "lon": -71.6197},
    {"name": "Concepcion", "country": "Chile", "continent": "South America", "lat": -36.8200, "lon": -73.0440},
    {"name": "La Serena", "country": "Chile", "continent": "South America", "lat": -29.9027, "lon": -71.2519},
    {"name": " Antofagasta", "country": "Chile", "continent": "South America", "lat": -23.6509, "lon": -70.3975},
    {"name": "Vina del Mar", "country": "Chile", "continent": "South America", "lat": -33.0246, "lon": -71.5518},
    {"name": "Puerto Montt", "country": "Chile", "continent": "South America", "lat": -41.4689, "lon": -72.9424},
    {"name": "Punta Arenas", "country": "Chile", "continent": "South America", "lat": -53.1638, "lon": -70.9171},
    {"name": "Temuco", "country": "Chile", "continent": "South America", "lat": -38.7396, "lon": -72.5904},
    {"name": "Iquique", "country": "Chile", "continent": "South America", "lat": -20.2133, "lon": -70.1503},
    {"name": "Chillan", "country": "Chile", "continent": "South America", "lat": -36.6065, "lon": -72.1036},
    {"name": "Rancagua", "country": "Chile", "continent": "South America", "lat": -34.1700, "lon": -70.7444},
    {"name": "Talca", "country": "Chile", "continent": "South America", "lat": -35.4264, "lon": -71.6665},
    {"name": "Arica", "country": "Chile", "continent": "South America", "lat": -18.4783, "lon": -70.1786},
    {"name": "Copiapo", "country": "Chile", "continent": "South America", "lat": -27.3666, "lon": -70.3314},
    {"name": "Osorno", "country": "Chile", "continent": "South America", "lat": -40.5739, "lon": -73.1336},

    # Madagascar
    {"name": "Antananarivo", "country": "Madagascar", "continent": "Africa", "lat": -18.8792, "lon": 47.5079},
    {"name": "Toamasina", "country": "Madagascar", "continent": "Africa", "lat": -18.1492, "lon": 49.4024},
    {"name": "Antsirabe", "country": "Madagascar", "continent": "Africa", "lat": -19.8719, "lon": 47.0333},
    {"name": "Mahajanga", "country": "Madagascar", "continent": "Africa", "lat": -15.7200, "lon": 46.3167},
    {"name": "Fianarantsoa", "country": "Madagascar", "continent": "Africa", "lat": -21.4536, "lon": 47.0840},
    {"name": "Toliara", "country": "Madagascar", "continent": "Africa", "lat": -23.3500, "lon": 43.6700},
    {"name": "Antsiranana", "country": "Madagascar", "continent": "Africa", "lat": -12.3231, "lon": 49.2918},
    {"name": "Sainte-Marie", "country": "Madagascar", "continent": "Africa", "lat": -17.0783, "lon": 49.8156},
    {"name": "Ifanadiana", "country": "Madagascar", "continent": "Africa", "lat": -21.5000, "lon": 47.5000},
    {"name": "Manakara", "country": "Madagascar", "continent": "Africa", "lat": -22.1333, "lon": 48.0167},
    {"name": "Vohemar", "country": "Madagascar", "continent": "Africa", "lat": -13.3500, "lon": 50.0000},
    {"name": "Ambanja", "country": "Madagascar", "continent": "Africa", "lat": -13.6833, "lon": 48.4500},
    {"name": "Andapa", "country": "Madagascar", "continent": "Africa", "lat": -14.6500, "lon": 49.6167},
    {"name": "Mampikony", "country": "Madagascar", "continent": "Africa", "lat": -16.0833, "lon": 47.6333},
    {"name": "Tsaratanana", "country": "Madagascar", "continent": "Africa", "lat": -16.7833, "lon": 47.6167},
    {"name": "Maroantsetra", "country": "Madagascar", "continent": "Africa", "lat": -15.4333, "lon": 49.7333},
]

# Build a fast lookup: group by country, sort by lat/lon
_CITY_DB_BY_COUNTRY: Dict[str, List[Dict]] = {}
for city in CITY_DATABASE:
    country = city["country"].lower()
    if country not in _CITY_DB_BY_COUNTRY:
        _CITY_DB_BY_COUNTRY[country] = []
    _CITY_DB_BY_COUNTRY[country].append(city)


def reverse_geocode(lat: float, lon: float, country_hint: Optional[str] = None) -> Dict:
    """
    Find the nearest known city from our database.

    Args:
        lat: Latitude
        lon: Longitude
        country_hint: Optional country to restrict the search

    Returns:
        Dict with name, country, continent, or closest match
    """
    if country_hint:
        candidates = _CITY_DB_BY_COUNTRY.get(country_hint.lower(), [])
    else:
        candidates = CITY_DATABASE

    if not candidates:
        return {"name": "Unknown", "country": "Unknown", "continent": "Unknown", "lat": lat, "lon": lon}

    best = None
    best_dist = float('inf')
    for city in candidates:
        d = haversine(lat, lon, city["lat"], city["lon"])
        if d < best_dist:
            best_dist = d
            best = city

    if best:
        return {"name": best["name"], "country": best["country"],
                "continent": best["continent"], "lat": lat, "lon": lon,
                "distance_km": round(best_dist, 2)}
    return {"name": "Unknown", "country": country_hint or "Unknown",
            "continent": "Unknown", "lat": lat, "lon": lon}


def enrich_geoclip_csv(csv_path: str) -> None:
    """
    Add city names to the geoclip CSV based on lat/lon reverse geocoding.
    Modifies the CSV in place, adding a 'nearest_city' column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    cities = []
    for _, row in df.iterrows():
        lat = row['lat']
        lon = row['lon']
        country = row.get('country', None)
        result = reverse_geocode(lat, lon, country)
        cities.append(result['name'])

    df['city'] = cities
    df.to_csv(csv_path, index=False)
    print(f"Enriched {len(df)} rows in {csv_path}")
    print(f"City distribution: {df['city'].value_counts().to_dict()}")


if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/home/user/data/geoclip/geoclip.csv"
    enrich_geoclip_csv(csv_path)
