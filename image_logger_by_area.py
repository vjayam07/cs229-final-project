import os
import osmnx as ox
ox.settings.use_cache = False
import geopandas as gpd
from shapely.geometry import box
import random
from random import sample
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from gc import collect
import threading

# Output directory for images
OUTPUT_DIR = "global_street_view_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Google API Key
API_KEY = "AIzaSyCZ-LTF1J9-t1n_wavLMXTcbpmJQuTYBHE"  # Replace with your actual API key

def check_street_view_availability(lat, lng):
    """
    Check if Street View imagery is available for a point.
    """
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "key": API_KEY}
    response = requests.get(metadata_url, params=params)
    data = response.json()
    return data.get("status") == "OK"

def download_street_view_image(lat, lng, filename):
    """
    Download a Street View image for a given point.
    """
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
#        print(f"Saved image: {filename}")
        return True
    else:
#        print(f"Failed to fetch image for ({lat}, {lng}): {response.status_code}")
        return False

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
    print(f'Generated {len(tiles)} tiles.')
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
            filename = os.path.join(OUTPUT_DIR, f"{country_name.replace(' ', '_')}_{lat:.6f}_{lng:.6f}.jpg")
            if check_street_view_availability(lat, lng) and download_street_view_image(lat, lng, filename):
                metadata.append({
                    "filename": filename,
                    "latitude": lat,
                    "longitude": lng,
                    "country": country_name
                })
#            else: print(f'No street view imagery at {lat, lng}')
        return metadata
    except Exception as e:
#        print(f"Error processing tile: {e}")
        return []

# def process_country(country_name, tile_size_km=50, sample_size_per_tile=20, area_threshold_km2=50000, tiles_per_km2=0.0001):
#     """
#     Process a country, handling small/medium countries without tiling.
#     """
#     print(f"Processing {country_name}...")
#     try:
#         # Geocode the country to get its polygon
#         country_gdf = ox.geocoder.geocode_to_gdf(country_name)
#         country_polygon = country_gdf.geometry.iloc[0]

#         # Calculate the area of the country in square kilometers
#         country_area = calculate_country_area(country_polygon)
#         print(f"{country_name} area: {country_area:.2f} km²")

#         metadata = []

#         if country_area <= area_threshold_km2:
#             # Small/Medium Country: Process the entire country polygon
#             print(f"{country_name} is a small/medium country. Processing without tiling.")

#             # Query the road network for the entire country
#             G = ox.graph_from_polygon(country_polygon, network_type="drive")
#             if G and len(G.edges) > 0:
#                 metadata.extend(process_tile(G, country_polygon, country_name, sample_size_per_tile))
#             else:
#                 print(f"No roads found in {country_name}.")
#         else:
#             # Large Country: Use proportional tiling
#             n_tiles = int(country_area * tiles_per_km2)
#             print(f"{country_name} is a large country. Sampling {n_tiles} tiles.")

#             # Generate all potential tiles for the country
#             tiles = generate_tiles(country_polygon, tile_size_km)

#             # Randomly select tiles until we find n_tiles with road networks
#             valid_tiles = []
#             attempts = 0
#             while len(valid_tiles) < n_tiles and attempts < len(tiles):
#                 # Select a random tile
#                 tile = sample(tiles, 1)[0]
#                 tiles.remove(tile)  # Avoid rechecking the same tile

#                 try:
#                     # Query the road network for the tile
#                     G = ox.graph_from_polygon(tile, network_type="drive")
#                     if G and len(G.edges) > 0:
#                         valid_tiles.append((tile, G))  # Store both tile and graph
#                 except Exception as e:
#                     print(f"Skipping tile: {e}")
#                 attempts += 1

#             # Process valid tiles
#             with ThreadPoolExecutor(max_workers=4) as executor:
#                 future_to_tile = {
#                     executor.submit(process_tile, G, tile, country_name, sample_size_per_tile): (tile, G)
#                     for tile, G in valid_tiles
#                 }
#                 for future in as_completed(future_to_tile):
#                     metadata.extend(future.result())

#         print(f"Finished processing {country_name}.")
#         return metadata
#     except Exception as e:
#         print(f"Error processing country {country_name}: {e}")
#         return []

# def process_country(country_name, tile_size_km=50, sample_size_per_tile=20, area_threshold_km2=50000, tiles_per_km2=0.0001):
#     """
#     Process a country, handling small/medium countries without tiling.
#     """
#     print(f"Processing {country_name}...")
#     try:
#         # Geocode the country to get its polygon
#         country_gdf = ox.geocoder.geocode_to_gdf(country_name)
#         country_polygon = country_gdf.geometry.iloc[0]

#         # Calculate the area of the country in square kilometers
#         country_area = calculate_country_area(country_polygon)
#         print(f"{country_name} area: {country_area:.2f} km²")

#         metadata = []

#         if country_area <= area_threshold_km2:
#             # Small/Medium Country: Process the entire country polygon
#             print(f"{country_name} is a small/medium country. Processing without tiling.")

#             # Query the road network for the entire country
#             G = ox.graph_from_polygon(country_polygon, network_type="drive")
#             if G and len(G.edges) > 0:
#                 metadata.extend(process_tile(G, country_polygon, country_name, sample_size_per_tile))
#             else:
#                 print(f"No roads found in {country_name}.")
#         else:
#             # Large Country: Use proportional tiling
#             n_tiles = int(country_area * tiles_per_km2)
#             print(f"{country_name} is a large country. Sampling {n_tiles} tiles.")

#             # Generate all potential tiles for the country
#             tiles = generate_tiles(country_polygon, tile_size_km)

#             # Randomly select tiles until we find n_tiles with road networks
#             processed_tiles = 0
#             attempts = 0
#             while processed_tiles < n_tiles and attempts < len(tiles):
#                 # Select a random tile
#                 tile = sample(tiles, 1)[0]
#                 tiles.remove(tile)  # Avoid rechecking the same tile

#                 try:
#                     # Query the road network for the tile
#                     G = ox.graph_from_polygon(tile, network_type="drive")
#                     if G and len(G.edges) > 0:
#                         # Process the tile immediately
#                         tile_metadata = process_tile(G, tile, country_name, sample_size_per_tile)
#                         metadata.extend(tile_metadata)

#                         # Clear the graph from memory
#                         del G
#                         import gc
#                         gc.collect()

#                         processed_tiles += 1
#                 except Exception as e:
#                     print(f"Skipping tile: {e}, have attempted {attempts} tiles.")
#                 attempts += 1

#         print(f"Finished processing {country_name}.")
#         return metadata
#     except Exception as e:
#         print(f"Error processing country {country_name}: {e}")
#         return []

from concurrent.futures import ThreadPoolExecutor, as_completed
from random import sample


def process_country(country_name, tile_size_km=50, sample_size_per_tile=20, area_threshold_km2=50000, tiles_per_km2=0.0001):
    """
    Process a country, handling small/medium countries without tiling. Parallel process large countries
    """
    total_num_images = 0
    print(f"Processing {country_name}...")
    try:
        # Geocode the country to get its polygon
        country_gdf = ox.geocoder.geocode_to_gdf(country_name)
        country_polygon = country_gdf.geometry.iloc[0]

        # Calculate the area of the country in square kilometers
        country_area = calculate_country_area(country_polygon)
        print(f"{country_name} area: {country_area:.2f} km²")

        if country_area <= area_threshold_km2:
            # Process small/medium countries without tiling
            print(f"{country_name} is a small/medium country. Processing without tiling.")
            G = ox.graph_from_polygon(country_polygon, network_type="drive")
            if G and len(G.edges) > 0:
                tile_metadata = process_tile(G, country_polygon, country_name, sample_size_per_tile)
                save_metadata(tile_metadata)
                total_num_images += len(tile_metadata)
            else:
                print(f"No roads found in {country_name}.")
        else:
            # Process large countries with tiling
            n_tiles = int(country_area * tiles_per_km2)
            print(f"{country_name} is a large country. Sampling {n_tiles} tiles.")
            tiles = generate_tiles(country_polygon, tile_size_km)
            random.shuffle(tiles)

            # # Define a helper function to query and process each tile
            # def process_single_tile(tile):
            #     try:
            #         G = ox.graph_from_polygon(tile, network_type="drive")
            #         if G and len(G.edges) > 0:
            #             tile_metadata = process_tile(G, tile, country_name, sample_size_per_tile)
            #             save_metadata(tile_metadata)
            #             return True
            #     except Exception as e:
            #         print(f"Skipping tile {attempts}: {e}.")
            #     finally:
            #         del G
            #         collect()
            #     return False

            # # Parallelize tile querying and processing
            # processed_tiles = 0
            # attempts = 0
            # with ThreadPoolExecutor(max_workers=4) as executor:
            #     futures = {
            #         executor.submit(process_single_tile, sample(tiles, 1)[0]): i
            #         for i in range(len(tiles))
            #     }

            #     for future in as_completed(futures):
            #         result = future.result()
            #         if result:
            #             processed_tiles += 1
            #         if processed_tiles % 50 == 0: print(f'Saved {processed_tiles} tiles.')
            #         if processed_tiles >= n_tiles:
            #             break  # Stop when enough tiles are processed


            processed_tiles = 0
            attempts = 0
            lock = threading.Lock()  # Lock to manage processed_tiles safely
            futures = []

            # Define a shared function to query and process tiles
            def process_single_tile(tile):
                nonlocal processed_tiles
                nonlocal attempts
                nonlocal total_num_images
                if processed_tiles >= n_tiles:
                    return False  # Stop if we've already processed enough tiles
                try:
                    G = ox.graph_from_polygon(tile, network_type="drive")
                    if G and len(G.edges) > 0:
                        # Process the tile and save metadata
                        tile_metadata = process_tile(G, tile, country_name, sample_size_per_tile)
                        save_metadata(tile_metadata)
                        total_num_images += len(tile_metadata)
                        del G
                        collect()
                        with lock:
                            processed_tiles += 1  # Update the counter safely
                        return True
                except Exception as e:
                    print(f"{processed_tiles}/{attempts}; Error processing tile: {e}")
                return False

            # Submit tasks to the ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=4) as executor:
                for tile in tiles:
                    if processed_tiles >= n_tiles:
                        break
                    futures.append(executor.submit(process_single_tile, tile))

                # Wait for futures and cancel unnecessary ones
                for future in as_completed(futures):
                    if processed_tiles >= n_tiles:
                        # Cancel remaining futures that haven't started
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        break

            print(f"Processed {processed_tiles}/{n_tiles} tiles for {country_name}")

        print(f"Finished processing {country_name}. Saved {total_num_images} images.")
        return
    except Exception as e:
        print(f"Error processing country {country_name}: {e}")
        return


def save_metadata(metadata, output_csv="global_street_view_metadata.csv"):
    """
    Save metadata to a CSV file.
    """
    if not metadata:
        return
    mode = "a" if os.path.exists(output_csv) else "w"
    header = not os.path.exists(output_csv)
    df = gpd.GeoDataFrame(metadata)
    df.to_csv(output_csv, mode=mode, index=False, header=header)
    print(f"Saved {len(metadata)} records to {output_csv}.")

if __name__ == "__main__":
    # List of countries to process
    test_countries_with_street_view_coverage = [
        "San Marino", "Denmark", "United States", "Canada", "France", "Germany", "Russia"
    ]
    countries_with_street_view_coverage = [
        # # Africa
        # "Botswana", "Ghana", "Kenya", "Lesotho", "Madagascar", "Nigeria", "Rwanda", "Senegal", "South Africa", "Uganda",
        # # Asia
        # "Bangladesh", "Hong Kong", "India", 
        # "Indonesia", "Israel", "Japan", "Jordan", "Laos", "Macau", "Malaysia",
        # "Mongolia", "Nepal", "Philippines", "Singapore", "South Korea", "Sri Lanka", "Taiwan", "Thailand", "Vietnam",
        # # Europe
        # "Albania", "Andorra", "Austria", "Belarus", "Belgium", "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus",
        # "Czech Republic", "Denmark", 
        # "Estonia", "Faroe Islands", "Finland", "France", "Germany", 
        # "Greece", "Hungary",
        # "Iceland", "Ireland", 
        # "Italy", "Kosovo", "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova",
        # "Monaco", "Montenegro", 
        # "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal", "Romania", "Russia",
        # "San Marino", "Serbia", "Slovakia", 
        #"Slovenia", 
        # "Spain", 
        # "Sweden", 
        #"Switzerland", 
        #"Ukraine", "United Kingdom",
        # "Vatican City",
        # North America
        # "Anguilla", "Antigua and Barbuda", "Aruba", "Bahamas", "Barbados", "Belize", "Bermuda", 
        # "Canada", 
        #"Cayman Islands",
        #"Costa Rica", "Curaçao", "Dominica", "Dominican Republic", "El Salvador", 
        #"Greenland", 
        #"Grenada", 
        #"Guadeloupe",
        #"Guatemala", "Haiti", 
        "Honduras", "Jamaica", "Martinique", "Mexico", "Nicaragua", "Panama", "Puerto Rico",
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
        process_country(
            country_name=country,
            tile_size_km=50,
            sample_size_per_tile=5,
            area_threshold_km2=10000,  # Adjust threshold as needed
            tiles_per_km2=0.0001       # Adjust scaling factor as needed
        )
        #save_metadata(metadata)
