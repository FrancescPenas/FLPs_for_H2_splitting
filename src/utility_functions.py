import pickle
import math
from pathlib import Path
import os
import csv
from IPython.display import display
# import numpy as np

def p(lst):
    # Prints each element of the list `lst` on a new line.
    print(*lst, sep='\n')

def pf(obj):
    # Pretty-prints the string representation of `obj` using pandas' DataFrame for better formatting.
    display(obj.to_string())

def find_index(lst, ch):
    # Returns indices of elements in `lst` that are strings starting with the specified character `ch`.
    return [i for i, item in enumerate(lst) if isinstance(item, str) and item.startswith(ch)]

# def find_index2(lst, ch):
#     """Returns indices of elements in `lst` that are exactly equal to the specified character `ch`."""
#     return [i for i, item in enumerate(lst) if item == ch]

def to_csv(data, csv_name, mode):
    # Writes `data` to a CSV file named `csv_name` using the specified `mode` ('w' for write, 'a' for append).
    if not isinstance(csv_name, str):
        raise ValueError('csv_name must be a string')
    if mode not in ('w', 'a'):
        raise ValueError("mode must be 'w' (write) or 'a' (append)")

    # Check if data is not empty before proceeding
    if not data:
        print("Warning: No data to write to CSV.")
        return
    
    path = Path(os.getcwd()) / csv_name
    with open(path, mode, newline='') as out:
        writer = csv.writer(out)
        if isinstance(data, list):
            writer.writerows(data if isinstance(data[0], list) else [[item] for item in data])
        else:
            writer.writerow([data])

def to_pkl(data, file_name='data.pkl'):
    # Saves `data` to a pickle file named `file_name`.
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)

def from_pkl(file_name):
    # Loads and returns data from a pickle file named `file_name`.
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def length(v):
    # Calculates the Euclidean length (magnitude) of a vector `v`.
    return math.sqrt(sum(coord**2 for coord in v))

def vector(b, e):
    # Calculates the vector from point `b` to point `e` by subtracting the coordinates of `b` from those of `e`.
    return [e_i - b_i for b_i, e_i in zip(b, e)]

# def distance(p0, p1):
#     """Calculates the Euclidean distance between points `p0` and `p1`."""
#     return length(vector(p0, p1))

def angle(v1, v2):
    # Calculates the angle in radians between two vectors `v1` and `v2` using the dot product and lengths of the vectors.
    dot_product = sum(a * b for a, b in zip(v1, v2))
    return math.acos(dot_product / (length(v1) * length(v2)))

def all_same(lst):
    # Checks if all elements in the list `lst` are the same.
    return len(set(lst)) == 1

# def divide_list(input_list, num_sublists):
#     """
#     Divides `input_list` into `num_sublists` of approximately equal length.
    
#     Returns a list of sublists.
#     """
#     avg_length = len(input_list) // num_sublists
#     remainder = len(input_list) % num_sublists
#     sublists = []
#     start = 0

#     for i in range(num_sublists):
#         end = start + avg_length + (1 if i < remainder else 0)
#         sublists.append(input_list[start:end])
#         start = end

#     return sublists

def normalize_dataframe(df):
    # Normalizes the values in a pandas DataFrame `df` to a range between 0 and 1 using min-max normalization.
    
    return (df - df.min()) / (df.max() - df.min())