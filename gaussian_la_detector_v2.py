import sys
import os
sys.path.append(r''+os.getcwd()+'\FLPs\Python\code')
os.chdir(r''+os.getcwd()+'\FLPs\Python\code')
from utility_functions_v2 import *

def gaussian_la_detector(file_name, nbo):
    # Check if nbo is missing or invalid
    if isinstance(nbo, str):
        return [f'NBO missing for {file_name}']
    if not nbo:
        return [f'Invalid or empty NBO data for {file_name}']

    # Initialize lists to collect LA data
    la_entries = []
    la_indices = []
    ener_list = []

    # Iterate through the nbo data
    for entry in nbo:
        if entry[1] == 'LP*(' and entry[3] == 'B':
            la_entries.append([entry, [int(entry[4]), int(float(entry[0]))]])
            la_indices.append(int(entry[4]))
            ener_list.append(float(entry[6]))

    # Handle cases based on collected LA data
    if not la_entries:
        return [f'No Lewis acid found for {file_name}']
    
    if len(la_entries) > 1:
        if all_same(la_indices):
            # Select the entry with the lowest energy
            min_energy_entry = min(la_entries, key=lambda x: float(x[0][6]))
            return min_energy_entry
        else:
            return [f'More than 1 possible Lewis acids found for {file_name}']
    
    return la_entries[0]

## Usage examples (commented out)
#x = from_pkl('input_data.pkl')
#nbo = x[3]['bicyclo_metriazole.log']
#y = gaussian_la_detector('bicyclo_metriazole.log', nbo)
