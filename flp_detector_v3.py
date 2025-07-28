from utility_functions_v2 import *
import os
import time
from gaussian_la_detector_v2 import gaussian_la_detector
from gaussian_lb_detector_v2 import gaussian_lb_detector

def flp_detector(pkl_inputfile, pkl_outfile='data_out'):
    start_time = time.time()
    error_list = []
    
    # Check if input file exists
    input_filename = f"{pkl_inputfile}.pkl"
    if input_filename not in os.listdir():
        error_list.append(f"No pkl file found with the name {input_filename}")
        p(error_list)
        to_csv(error_list, 'flp_detector_error_list.csv', 'w')
        return None
    
    # Load data from input file
    x = from_pkl(input_filename)
    file_name_list, coordinates_dict, nbo_dict, connectivity_dict = x[1], x[2], x[3], x[4]

    la_list, lb_list, dist_list = [], [], []

    # Process each file entry
    for file_name in file_name_list:
        la, lb = 0, 0
        coordinates, nbo, connectivity = coordinates_dict[file_name], nbo_dict[file_name], connectivity_dict[file_name]
        
        # Run la detector
        la = gaussian_la_detector(file_name, nbo)
        if isinstance(la[0], str):  # Handle error in la detector
            error_list.append(f"{file_name}: {la[0]}")
            la_list.append(la[0])
            lb_list.append(la[0])
            dist_list.append(la[0])
            continue  # Skip lb detector if la failed
        
        # Store la result and run lb detector
        la_list.append(la[1])
        lb = gaussian_lb_detector(file_name, coordinates, nbo, connectivity, la[1][0])
        
        if isinstance(lb[0], str):  # Handle error in lb detector
            error_list.append(f"{file_name}: {lb[0]}")
            lb_list.append(lb[0])
            dist_list.append(lb[0])
        else:
            lb_list.append(lb[0][1])
            dist_list.append(lb[0][2])

    # Create dictionaries of the extracted data
    la_dict = dict(zip(file_name_list, la_list))
    lb_dict = dict(zip(file_name_list, lb_list))
    dist_dict = dict(zip(file_name_list, dist_list))

    y = [la_dict, lb_dict, dist_dict]

    # Save error list and results
    if error_list:
        p(error_list)
        to_csv(error_list, 'error_list.csv', 'w')
    
    to_pkl(y, f"{pkl_outfile}.pkl")
    
    execution_time = time.time() - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")
    return y

# y = flp_detector('input_data', 'data_flp')
# x = from_pkl('input_data.pkl')
# file_name_list = x[1]
# nbo_dict = x[3]
# connectivity_dict = x[4]