import sys
import os
#sys.path.append(r''+os.getcwd()+'\FLPs\Python\code')
#os.chdir(r''+os.getcwd()+'\FLPs\Python\code')
from utility_functions import *
from gaussian_reader import gaussian_reader
import pandas as pd

def gaussian_nat_charges_extractor(file_name, total_lines):
    opt_check = find_index(total_lines, ' Summary of Natural Population Analysis:                  ')
    if not opt_check:
        print('No Natural charges found for ' + file_name)
        return []
    
    last_pos = opt_check.pop()
    nat_char = []
    n = last_pos + 6
    
    while True:
        current_line = total_lines[n]
        if '=============' in current_line:
            break
        try:
            current_line = current_line.split()
            nat_char.append([current_line[0], current_line[1], float(current_line[2])])
        except (IndexError, ValueError) as e:
            print(f"Error parsing line {n} in file {file_name}: {e}")
            break
        n += 1
    
    # Create DataFrame
    columns = ['Atom', 'No', 'Natural Charge']
    nat_char_df = pd.DataFrame(nat_char, columns=columns)

    return nat_char_df

## Load the total lines from the Gaussian log file
#total_lines = gaussian_reader('cl_dibenzofuran.log', 'input')

## Extract the natural charges
#nat_char = gaussian_nat_charges_extractor('cl_dibenzofuran.log', total_lines)

## Print the extracted charges
#print(nat_char)