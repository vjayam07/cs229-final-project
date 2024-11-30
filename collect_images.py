import os
import requests
import pandas as pd
from geopy.geocoders import Nominatim

# Your Google API key
API_KEY = "AIzaSyCZ-LTF1J9-t1n_wavLMXTcbpmJQuTYBHE"

# Directory to save images
OUTPUT_DIR = "valid_street_view_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Predefined POIs (latitude, longitude)
poi_coordinates = [
    (37.7749, -122.4194),  # San Francisco, USA
    (48.8566, 2.3522),     # Paris, France
    (35.6895, 139.6917),   # Tokyo, Japan
    (51.5074, -0.1278),    # London, UK
]

# Function to check Street View availability
def check_street_view_availability(lat, lng):
    metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    params = {"location": f"{lat},{lng}", "key": API_KEY}
    response = requests.get(metadata_url, params=params)
    data = response.json()
    return data.get("status") == "OK"

# Function to download a Street View image
def download_street_view_image(lat, lng, filename):
    base_url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": "640x640",  # Image resolution
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

# Metadata storage
metadata = []

# Loop through known valid POIs
for i, (lat, lng) in enumerate(poi_coordinates):
    if check_street_view_availability(lat, lng):
        country = "Unknown"
        filename = os.path.join(OUTPUT_DIR, f"poi_{i}.jpg")
        if download_street_view_image(lat, lng, filename):
            metadata.append({"filename": filename, "latitude": lat, "longitude": lng, "country": country})

# Save metadata to CSV
metadata_df = pd.DataFrame(metadata)
csv_path = os.path.join(OUTPUT_DIR, "metadata.csv")
metadata_df.to_csv(csv_path, index=False)
print(f"Metadata saved to {csv_path}")
