import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
os.chdir(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
from re import I
from utility_functions import *
import time
import os
from sys import stdout
import csv
from gaussian_esp_charges_extractor import gaussian_esp_charges_extractor
from gaussian_nat_charges_extractor import gaussian_nat_charges_extractor
from gaussian_reader import gaussian_reader
from gaussian_coordinates_extractor import gaussian_coordinates_extractor
from gaussian_nbo_extractor import gaussian_nbo_extractor
from gaussian_connectivity_extractor import gaussian_connectivity_extractor

def data_extractor(input_dir, esp_charges_dir='no_esp'):
    start_time = time.time()
    path = os.getcwd()
    current_file = 0
    error_list = []
    coordinates_list = []
    nbo_list = []
    connectivity_list = []
    nat_char_list = []
    
    dir_path = os.path.join(path, input_dir)
    list_files = os.listdir(dir_path)
    list_files.sort()  # Ensure consistent ordering
    
    while current_file < len(list_files):
        # Reader
        file_name = list_files[current_file]
        total_lines = gaussian_reader(file_name, input_dir)
        
        # Update progress on the same line
        comp = 100 * (current_file + 1) / len(list_files)
        stdout.write("\r%d%% completed " % comp)
        stdout.flush()
        print(file_name)
        
        # Coordinates extractor
        coordinates = gaussian_coordinates_extractor(file_name, total_lines)
        if isinstance(coordinates, str):  # If there's an error, add to error list
            error_list.append(coordinates)
        coordinates_list.append(coordinates)
        
        # NBO extractor
        nbo = gaussian_nbo_extractor(file_name, total_lines)
        if isinstance(nbo, str):
            error_list.append(nbo)
        nbo_list.append(nbo)
        
        # Connectivity extractor
        connectivity = gaussian_connectivity_extractor(file_name, total_lines) if not isinstance(coordinates, str) else coordinates
        if isinstance(connectivity, str):
            error_list.append(connectivity)
        connectivity_list.append(connectivity)
        
        # NAT charges extractor
        nat_char = gaussian_nat_charges_extractor(file_name, total_lines)
        nat_char_list.append(nat_char)
        
        current_file += 1
    
    # ESP charges extraction if specified
    if esp_charges_dir != 'no_esp':
        esp_charges_list = []
        current_file = 0
        dir_path_esp = os.path.join(path, esp_charges_dir)
        list_files_esp = os.listdir(dir_path_esp)
        list_files_esp.sort()  # Ensure consistent ordering
        
        while current_file < len(list_files_esp):
            # Reader
            file_name = list_files_esp[current_file]
            total_lines = gaussian_reader(file_name, esp_charges_dir)
            
            # ESP charges extractor
            esp_charges = gaussian_esp_charges_extractor(file_name, total_lines)
            esp_charges_list.append(esp_charges)
            current_file += 1
            
        esp_charges_dict = dict(zip(list_files_esp, esp_charges_list))
    
    # Dictionaries of extracted data
    coordinates_dict = dict(zip(list_files, coordinates_list))
    nbo_dict = dict(zip(list_files, nbo_list))
    connectivity_dict = dict(zip(list_files, connectivity_list))
    nat_char_dict = dict(zip(list_files, nat_char_list))
    
    # Compile results into the output variable
    x = [error_list, list_files, coordinates_dict, nbo_dict, connectivity_dict, nat_char_dict]
    if esp_charges_dir != 'no_esp':
        x.append(esp_charges_dict)

    # Print errors if any
    if error_list:
        print("Errors encountered:")
        print(error_list)
        # Save error list to CSV
        to_csv(error_list, 'data_extractor_error_list.csv', 'w')
    
    # Save data to pickle
    to_pkl(x, input_dir + '_data.pkl')
    
    # Print execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nExecution Time: {execution_time:.2f} seconds")

    # Return the results and number of processed files
    return [x, len(list_files)]

# Example usage:

x = data_extractor('input_ts')
