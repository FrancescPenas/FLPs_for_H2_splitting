import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
os.chdir(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
from utility_functions import *
import numpy as np
import pandas as pd

def dist_calc(coordinates_df, x, y):
    # Validate indices (assuming x and y are 1-based indices)
    if x <= 0 or y <= 0 or x > len(coordinates_df) or y > len(coordinates_df):
        raise ValueError("Invalid indices: x and y must be within the valid range.")
    
    # Extract coordinates for x and y (adjusting for 0-based indexing)
    # Ensure the coordinates are converted to float for accurate calculations
    x_coord = coordinates_df.iloc[x - 1][['X', 'Y', 'Z']].astype(float).values
    y_coord = coordinates_df.iloc[y - 1][['X', 'Y', 'Z']].astype(float).values
    
    # Calculate the Euclidean distance using numpy
    dist = np.linalg.norm(x_coord - y_coord)
    
    return dist


# file_name = 'bicyclo_metriazole.log'
# x = from_pkl('input_data.pkl')
# coordinates = x[2][file_name]
# dist = dist_calc(coordinates, 1, 2)