import json
import random

from datetime import datetime, timedelta

# Expanded mapping for all major locations in your dataset
LOCATION_COORDS = {
    "Banff, Canada": (51.1784, -115.5708),
    "Malibu, USA": (34.0259, -118.7798),
    "Toronto, Canada": (43.651070, -79.347015),
    "Queenstown, New Zealand": (-45.0312, 168.6626),
    "Kilimanjaro, Tanzania": (-3.0674, 37.3556),
    "Santorini, Greece": (36.3932, 25.4615),
    "Bali, Indonesia": (-8.3405, 115.0920),
    "Paris, France": (48.8566, 2.3522),
    "Rome, Italy": (41.9028, 12.4964),
    "Sydney, Australia": (-33.8688, 151.2093),
    "Cape Town, South Africa": (-33.9249, 18.4241),
    "Zurich, Switzerland": (47.3769, 8.5417),
    "Tokyo, Japan": (35.6895, 139.6917),
    "New York, USA": (40.7128, -74.0060),
    "London, UK": (51.5074, -0.1278),
    # Add more as needed
}

# Helper to get base coordinates for a location string
def get_base_coords(location):
    for key, coords in LOCATION_COORDS.items():
        if key in location:
            return coords
    return (0.0, 0.0)

# Add a small random offset to base coordinates
# Offset is up to ~0.02 degrees (~2km)
def randomize_coords(base_lat, base_lng):
    lat_offset = random.uniform(-0.02, 0.02)
    lng_offset = random.uniform(-0.02, 0.02)
    return {"lat": round(base_lat + lat_offset, 6), "lng": round(base_lng + lng_offset, 6)}

def random_dates(num=3, year=2025):
    base = datetime(year, 9, 1)
    return [
        (base + timedelta(days=random.randint(0, 90))).strftime("%Y-%m-%d")
        for _ in range(num)
    ]

with open("datasets/property_listings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for prop in data["properties"]:
    base_lat, base_lng = get_base_coords(prop["location"])
    prop["coordinates"] = randomize_coords(base_lat, base_lng)
    prop["booked_dates"] = random_dates(random.randint(2, 6))

with open("datasets/property_listings.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Property listings updated with coordinates and random booked_dates.")
