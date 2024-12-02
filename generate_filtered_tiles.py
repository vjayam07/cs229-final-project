import os
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box

# Directory to save tile files
TILE_DIR = "country_tiles"
os.makedirs(TILE_DIR, exist_ok=True)

def generate_and_save_tiles(country_name, tile_size_km=50):
    """
    Generate tiles for a country and save them as a GeoJSON file.
    """
    try:
        # Geocode the country to get its polygon
        country_gdf = ox.geocoder.geocode_to_gdf(country_name)
        country_polygon = country_gdf.geometry.iloc[0]

        # Generate tiles
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

        # Save tiles as a GeoDataFrame
        tile_gdf = gpd.GeoDataFrame(geometry=tiles, crs="EPSG:4326")
        tile_file = os.path.join(TILE_DIR, f"{country_name.replace(' ', '_')}_tiles.geojson")
        tile_gdf.to_file(tile_file, driver="GeoJSON")
        print(f"Saved {len(tiles)} tiles for {country_name} to {tile_file}")
    except Exception as e:
        print(f"Error generating tiles for {country_name}: {e}")

def filter_tiles_with_roads(country_name):
    """
    Filter tiles for a country to keep only those with roads.
    """
    try:
        # Load tiles from the saved GeoJSON file
        tile_file = os.path.join(TILE_DIR, f"{country_name.replace(' ', '_')}_tiles.geojson")
        tile_gdf = gpd.read_file(tile_file)

        # Keep tiles that contain roads
        valid_tiles = []
        for tile in tile_gdf.geometry:
            try:
                # Query the road network within the tile
                G = ox.graph_from_polygon(tile, network_type="drive")
                if G and len(G.nodes) > 0:
                    valid_tiles.append(tile)
            except Exception as e:
                print(f"No roads in tile: {tile.bounds}, Error: {e}")

        # Save the filtered tiles
        filtered_tile_gdf = gpd.GeoDataFrame(geometry=valid_tiles, crs="EPSG:4326")
        filtered_tile_file = os.path.join(TILE_DIR, f"{country_name.replace(' ', '_')}_filtered_tiles.geojson")
        filtered_tile_gdf.to_file(filtered_tile_file, driver="GeoJSON")
        print(f"Saved {len(valid_tiles)} filtered tiles for {country_name} to {filtered_tile_file}")
    except Exception as e:
        print(f"Error filtering tiles for {country_name}: {e}")

def process_country(country_name, tile_size_km=50):
    """
    Generate and filter tiles for a single country.
    """
    print(f"Processing {country_name}...")
    generate_and_save_tiles(country_name, tile_size_km)
    print(f"Filtering tiles in {country_name}...")
    filter_tiles_with_roads(country_name)

def process_countries(country_list, tile_size_km=50):
    """
    Process a list of countries to generate and filter tiles.
    """
    for country in country_list:
        process_country(country, tile_size_km)

if __name__ == "__main__":
    # List of countries to process
    countries_with_street_view_coverage = [
        "United States", "Canada", "Brazil", "France", "Germany", "India", "Japan", "Australia"
        # Add more countries as needed
    ]

    # Process each country
    process_countries(countries_with_street_view_coverage, tile_size_km=50)
