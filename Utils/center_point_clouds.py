import os
import pandas as pd
import numpy as np

def center_point_clouds_in_folder(input_folder, output_folder=None):
    if output_folder is None:
        output_folder = input_folder  # Save centered files in same folder

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(input_folder, filename)

            try:
                df = pd.read_csv(file_path, delim_whitespace=True)
                points = df[['x', 'y', 'z']].values
                centroid = points.mean(axis=0)
                df[['x', 'y', 'z']] = points - centroid

                output_filename = f"{os.path.splitext(filename)[0]}_centered.txt"
                output_path = os.path.join(output_folder, output_filename)

                df.to_csv(output_path, sep=' ', index=False, float_format='%.10f')
                print(f"Processed: {filename} -> {output_filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Example usage
input_directory = './Clouds'  # Replace with your folder path
center_point_clouds_in_folder(input_directory)
