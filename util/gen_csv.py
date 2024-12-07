import os
import csv

def generate_csv_from_filenames(folder_path, output_csv):
    """
    Generates a CSV from filenames in a folder.
    
    Each filename should be in the format: Country_Latitude_Longitude.jpg.
    
    Parameters:
        folder_path (str): Path to the folder containing the image files.
        output_csv (str): Path to the output CSV file.
    """
    # Prepare the output CSV file
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Filename", "Country", "Latitude", "Longitude"])  # Header row

        # Iterate through the files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"):  # Only process JPG files
                try:
                    # Remove file extension and split from the right
                    base_name = filename.replace(".jpg", "")
                    parts = base_name.rsplit("_", 2)  # Split into three parts from the right
                    if len(parts) != 3:
                        print(f"Skipping file with unexpected format: {filename}")
                        continue
                    
                    country = parts[0]
                    lat = parts[1]
                    lng = parts[2]
                    writer.writerow([filename, country, lat, lng])
                except ValueError:
                    print(f"Skipping file with unexpected format: {filename}")

    print(f"CSV file generated at: {output_csv}")

# Example usage
folder_path = "global_street_view_images"  # Replace with your folder path
output_csv = "street_view_metadata.csv"   # Replace with your desired output CSV path
generate_csv_from_filenames(folder_path, output_csv)
