import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
os.chdir(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
import pandas as pd
from utility_functions import *
from gaussian_reader import gaussian_reader

def gaussian_coordinates_extractor(file_name, total_lines):
    try:
        # Check if optimization has converged
        opt_check = find_index(total_lines, '                           !   Optimized Parameters   !')
        if not opt_check:
            return f"{file_name} no converged"
        
        # Find the last occurrence of 'Input orientation:'
        input_orientation_positions = find_index(total_lines, '                          Input orientation:')
        if not input_orientation_positions:
            return f"No coordinates found for {file_name}"
        
        # Extract coordinates from the last 'Input orientation:' section
        last_pos = input_orientation_positions[-1]
        coordinates = []
        line_index = last_pos + 5  # Coordinates start 5 lines after 'Input orientation:'
        
        while True:
            line = total_lines[line_index]
            if line.startswith(' ---'):
                break
            coordinates.append(line.split())
            line_index += 1
        
        # Create DataFrame
        columns = ['Center number', 'Atomic Number', 'Atom Type', 'X', 'Y', 'Z']
        coordinates_df = pd.DataFrame(coordinates, columns=columns)
        
        return coordinates_df
    except Exception as e:
        return f"An error occurred: {e}"

# Example Usage
#total_lines = gaussian_reader('cf3_dibenzofuran.log', 'input')
#coordinates = gaussian_coordinates_extractor('cf3_dibenzofuran.log', total_lines)
#print(coordinates)
