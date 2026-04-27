import numpy as np

def dist_calc(coordinates_df, x, y):
    # Indices validation (assuming x and y are 1-based indices)
    if x <= 0 or y <= 0 or x > len(coordinates_df) or y > len(coordinates_df):
        raise ValueError("Invalid indices: x and y must be within the valid range.")
    
    # Coordinates extraction for x and y (adjusting for 0-based indexing)
    x_coord = coordinates_df.iloc[x - 1][['X', 'Y', 'Z']].astype(float).values
    y_coord = coordinates_df.iloc[y - 1][['X', 'Y', 'Z']].astype(float).values
    
    # Euclidean distance calculation using numpy
    dist = np.linalg.norm(x_coord - y_coord)
    
    return dist