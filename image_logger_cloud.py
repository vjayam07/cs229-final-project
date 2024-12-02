import os
import osmnx as ox
ox.settings.use_cache = False
import geopandas as gpd
from shapely.geometry import box
from random import sample
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
import pandas as pd
from io import StringIO

# Google Cloud Storage bucket name
BUCKET_NAME = "streetview_images"  # Replace with your GCS bucket name
IMAGE_FOLDER = "global_street_view_images"  # Folder name in GCS

# Google Street View API Key
API_KEY = "AIzaSyCZ-LTF1J9-t1n_wavLMXTcbpmJQuTYBHE"


def upload_to_gcs(bucket_name, file_data, destination_blob_name):
    """
    Upload file data directly to Google Cloud Storage.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(file_data)
    print(f"Uploaded to GCS: {destination_blob_name}")


def download_street_view_image(lat, lng, bucket_name, folder_name):
    """
    Download a Street View image and upload it to a specified folder in GCS.
    """
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",
        "location": f"{lat},{lng}",
        "key": API_KEY
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Save the image in the specified folder
        filename = f"{lat:.6f}_{lng:.6f}.jpg"
        destination_blob_name = f"{folder_name}/{filename}"
        upload_to_gcs(bucket_name, response.content, destination_blob_name)
        return True
    else:
        print(f"Failed to fetch image for ({lat}, {lng}): {response.status_code}")
        return False


def save_metadata(metadata, bucket_name, output_csv="global_street_view_metadata.csv"):
    """
    Append metadata to an existing CSV in GCS or create a new one if it doesn't exist.
    """
    if not metadata:
        return

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(output_csv)

    # Create a GeoDataFrame for the new metadata
    new_df = gpd.GeoDataFrame(metadata)

    if blob.exists():
        # If the file exists in GCS, download it and append the new data
        existing_data = blob.download_as_text()
        existing_df = pd.read_csv(StringIO(existing_data))
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file doesn't exist, use the new data as the combined data
        combined_df = new_df

    # Save the combined data back to GCS
    csv_data = combined_df.to_csv(index=False)
    upload_to_gcs(bucket_name, csv_data, output_csv)
    print(f"Metadata appended and uploaded to GCS: {output_csv}")


def generate_tiles(country_polygon, tile_size_km=50):
    """
    Generate a list of tiles for a given country polygon.
    """
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


def calculate_country_area(country_polygon):
    """
    Calculate the area of the country polygon in square kilometers.
    """
    # Convert the polygon to a GeoSeries to use the to_crs method
    country_geo = gpd.GeoSeries([country_polygon], crs="EPSG:4326")
    # Reproject to a metric CRS and calculate the area
    area_km2 = country_geo.to_crs(epsg=3395).area.iloc[0] / 10**6  # EPSG 3395 is a metric projection
    return area_km2


def process_tile(G, tile, country_name, sample_size_per_tile):
    """
    Process a single tile or country polygon: sample points from the road network.
    """
    try:
        # Ensure the graph is undirected and in a projected CRS
        G = ox.project_graph(G)
        G = G.to_undirected()

        # Sample random points on the road network
        sampled_points = ox.utils_geo.sample_points(G, n=sample_size_per_tile)
        sampled_points = sampled_points.to_crs(epsg=4326)

        metadata = []
        for _, point in sampled_points.items():
            lat, lng = point.y, point.x
            if download_street_view_image(lat, lng, BUCKET_NAME, IMAGE_FOLDER):
                metadata.append({
                    "latitude": lat,
                    "longitude": lng,
                    "country": country_name
                })
        return metadata
    except Exception as e:
        print(f"Error processing tile: {e}")
        return []


def process_country(country_name, tile_size_km=50, sample_size_per_tile=20, area_threshold_km2=50000, tiles_per_km2=0.0001):
    """
    Process a country, handling small/medium countries without tiling.
    """
    print(f"Processing {country_name}...")
    try:
        # Geocode the country to get its polygon
        country_gdf = ox.geocoder.geocode_to_gdf(country_name)
        country_polygon = country_gdf.geometry.iloc[0]

        # Calculate the area of the country in square kilometers
        country_area = calculate_country_area(country_polygon)
        print(f"{country_name} area: {country_area:.2f} km²")

        metadata = []

        if country_area <= area_threshold_km2:
            # Small/Medium Country: Process the entire country polygon
            print(f"{country_name} is a small/medium country. Processing without tiling.")

            # Query the road network for the entire country
            G = ox.graph_from_polygon(country_polygon, network_type="drive")
            if G and len(G.edges) > 0:
                metadata.extend(process_tile(G, country_polygon, country_name, sample_size_per_tile))
            else:
                print(f"No roads found in {country_name}.")
        else:
            # Large Country: Use proportional tiling
            n_tiles = int(country_area * tiles_per_km2)
            print(f"{country_name} is a large country. Sampling {n_tiles} tiles.")

            # Generate all potential tiles for the country
            tiles = generate_tiles(country_polygon, tile_size_km)

            # Randomly select tiles until we find n_tiles with road networks
            valid_tiles = []
            attempts = 0
            while len(valid_tiles) < n_tiles and attempts < len(tiles):
                # Select a random tile
                tile = sample(tiles, 1)[0]
                tiles.remove(tile)  # Avoid rechecking the same tile

                try:
                    # Query the road network for the tile
                    G = ox.graph_from_polygon(tile, network_type="drive")
                    if G and len(G.edges) > 0:
                        valid_tiles.append((tile, G))  # Store both tile and graph
                except Exception as e:
                    print(f"Skipping tile: {e}")
                attempts += 1

            # Process valid tiles
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_tile = {
                    executor.submit(process_tile, G, tile, country_name, sample_size_per_tile): (tile, G)
                    for tile, G in valid_tiles
                }
                for future in as_completed(future_to_tile):
                    metadata.extend(future.result())

        print(f"Finished processing {country_name}.")
        return metadata
    except Exception as e:
        print(f"Error processing country {country_name}: {e}")
        return []


if __name__ == "__main__":
    # List of countries to process
    test_countries_with_street_view_coverage = [
        "San Marino", "Denmark", "United States", "Canada", "France", "Germany", "Russia"
    ]
    countries_with_street_view_coverage = [
        # Africa
        "Botswana", "Ghana", "Kenya", "Lesotho", "Madagascar", "Nigeria", "Rwanda", "Senegal", "South Africa", "Uganda",
        # Asia
        "Bangladesh", "Hong Kong", "India", "Indonesia", "Israel", "Japan", "Jordan", "Laos", "Macau", "Malaysia",
        "Mongolia", "Nepal", "Philippines", "Singapore", "South Korea", "Sri Lanka", "Taiwan", "Thailand", "Vietnam",
        # Europe
        "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        "Czech Republic", "Denmark", "Estonia", "Faroe Islands", "Finland", "France", "Germany", "Greece", "Hungary",
        "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova",
        "Monaco", "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia",
        "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "Switzerland", "Ukraine", "United Kingdom",
        "Vatican City",
        # North America
        "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize", "Bermuda", "Canada", "Cayman Islands",
        "Costa Rica", "Curaçao", "Dominica", "Dominican Republic", "El Salvador", "Greenland", "Grenada", "Guadeloupe",
        "Guatemala", "Haiti", "Honduras", "Jamaica", "Martinique", "Mexico", "Nicaragua", "Panama", "Puerto Rico",
        "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Sint Maarten", "Trinidad and Tobago",
        "United States", "U.S. Virgin Islands",
        # Oceania
        "Australia", "Fiji", "French Polynesia", "Guam", "New Caledonia", "New Zealand", "Northern Mariana Islands",
        "Papua New Guinea", "Samoa",
        # South America
        "Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Paraguay", "Peru", "Uruguay"
    ]

    # Process each country
    for country in countries_with_street_view_coverage:
        metadata = process_country(
            country_name=country,
            tile_size_km=50,
            sample_size_per_tile=20,
            area_threshold_km2=50000,  # Adjust threshold as needed
            tiles_per_km2=0.0001       # Adjust scaling factor as needed
        )
        save_metadata(metadata, BUCKET_NAME)
