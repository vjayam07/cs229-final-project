import os
import osmnx as ox
import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep

# Google API Key
API_KEY = "AIzaSyCZ-LTF1J9-t1n_wavLMXTcbpmJQuTYBHE"

# Output directory for images
OUTPUT_DIR = "global_street_view_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CSV file to store metadata
CSV_FILE = "global_street_view_metadata.csv"

# Function to download a Street View image
def download_street_view_image(lat, lng, filename):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{lat},{lng}",
        "key": API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Saved image: {filename}")
        return True
    else:
        print(f"Failed to fetch image for ({lat}, {lng}): {response.status_code}")
        return False

# Function to verify Street View availability
def check_street_view_availability(lat, lng):
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "key": API_KEY}
    try:
        response = requests.get(metadata_url, params=params)
        data = response.json()
        if data.get("status") != "OK":
            print(f"Metadata check failed for ({lat}, {lng}): {data}")
        return data.get("status") == "OK"
    except Exception as e:
        print(f"Error checking metadata for ({lat}, {lng}): {e}")
        return False

# Function to generate tiles for a country
def generate_tiles(country_polygon, tile_size_km):
    bounds = country_polygon.bounds
    minx, miny, maxx, maxy = bounds
    tile_size_deg = tile_size_km / 111  # Approximation: 1 degree ≈ 111 km

    tiles = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            tile = box(x, y, x + tile_size_deg, y + tile_size_deg)
            if country_polygon.intersects(tile):
                tiles.append(tile)
            y += tile_size_deg
        x += tile_size_deg

    return tiles

# Function to process a tile and sample random points
def process_tile(tile, country_name, sample_size_per_tile):
    try:
        # Fetch road network within the tile
        G = ox.graph_from_polygon(tile, network_type="drive")
        if G is None or len(G.nodes) == 0:
            print(f"No road network data for tile: {tile.bounds}")
            return []

        G = ox.project_graph(G)
        G = G.to_undirected()

        # Sample random points
        sampled_points = ox.utils_geo.sample_points(G, n=sample_size_per_tile)
        sampled_points = sampled_points.to_crs(epsg=4326)

        metadata = []
        for _, point in sampled_points.items():
            lat, lng = point.y, point.x
            filename = os.path.join(OUTPUT_DIR, f"{country_name.replace(' ', '_')}_{lat:.6f}_{lng:.6f}.jpg")
            if check_street_view_availability(lat, lng) and download_street_view_image(lat, lng, filename):
                metadata.append({
                    "filename": filename,
                    "latitude": lat,
                    "longitude": lng,
                    "country": country_name
                })
        return metadata
    except Exception as e:
        print(f"Error processing tile for {country_name}: {e}")
        return []

# Function to process a country: generate tiles and parallelize tile processing
def process_country(country_name, tile_size_km, n_tiles, sample_size_per_tile):
    print(f"Processing {country_name}...")

    try:
        country_polygon = ox.geocoder.geocode_to_gdf(country_name).geometry.iloc[0]
        tiles = generate_tiles(country_polygon, tile_size_km)
        print(f"Generated {len(tiles)} tiles for {country_name}.")

        # Randomly select tiles to process
        selected_tiles = np.random.choice(tiles, size=min(n_tiles, len(tiles)), replace=False)

        metadata = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_tile = {executor.submit(process_tile, tile, country_name, sample_size_per_tile): tile for tile in selected_tiles}
            for future in as_completed(future_to_tile):
                metadata.extend(future.result())

        return metadata
    except Exception as e:
        print(f"Error processing country {country_name}: {e}")
        return []

# Function to save metadata to CSV (append mode)
def save_metadata(metadata):
    if not metadata:
        return
    mode = "a" if os.path.exists(CSV_FILE) else "w"
    header = not os.path.exists(CSV_FILE)
    df = pd.DataFrame(metadata)
    df.to_csv(CSV_FILE, mode=mode, index=False, header=header)
    print(f"Saved {len(metadata)} records to {CSV_FILE}.")

# Function to process all countries globally in parallel
def process_global(tile_size_km=50, n_tiles=10, sample_size_per_tile=20):
    countries = [
        "United States", "India", "Brazil", "Germany", "Australia", "Canada", "Russia",
        "China", "Japan", "South Africa", "United Kingdom", "France", "Mexico"
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_country = {
            executor.submit(process_country, country, tile_size_km, n_tiles, sample_size_per_tile): country for country in countries
        }
        for future in as_completed(future_to_country):
            metadata = future.result()
            save_metadata(metadata)

# Main execution
if __name__ == "__main__":
    process_global(tile_size_km=50, n_tiles=2, sample_size_per_tile=5)
