import sys
import os
#sys.path.append(r''+os.getcwd()+'\FLPs\Python\code')
#os.chdir(r''+os.getcwd()+'\FLPs\Python\code')
import re
from utility_functions_v2 import *
from gaussian_reader_v2 import gaussian_reader

def gaussian_connectivity_extractor(file_name, total_lines):
    """
    Extracts atom connectivity information from a Gaussian log file starting from the optimized parameters section.
    
    Parameters:
    - file_name: The name of the Gaussian log file.
    - total_lines: A list of lines read from the Gaussian log file.
    
    Returns:
    - A list of connectivity data (list of lists of integers), or a message if connectivity is not found.
    """
    # Find the index of the optimized parameters section
    opt_check = find_index(total_lines, '                           !   Optimized Parameters   !')
    
    # If the optimized parameters section isn't found, return a message
    if not opt_check:
        return f'No connectivity found for {file_name}'
    
    opt_check = opt_check[0]  # Take the first match (assume there's only one)
    connectivity = []
    
    # Start from 5 lines after the optimized parameters section
    n = opt_check + 5
    
    # Iterate over lines to extract connectivity
    while n < len(total_lines):
        current_line = total_lines[n].strip()
        
        # Check for the end of the connectivity section (marked by '---')
        if current_line.startswith('---'):
            break
        
        # Extract the connectivity (assume it's the third column)
        # Use a regular expression to find numbers in the third column
        try:
            connectivity_line = current_line.split()[2]  # Assuming connectivity is in the third column
            connectivity_data = [int(s) for s in re.findall(r'\d+', connectivity_line)]
            connectivity.append(connectivity_data)
        except IndexError:
            # Handle case where the expected data is missing or malformed
            print(f"Warning: Unable to extract connectivity data from line: {current_line}")
        
        # Move to the next line
        n += 1
    
    if connectivity:
        return connectivity
    else:
        return f'No connectivity found for {file_name}'

## Example usage:
## Load the log file into `total_lines`
#total_lines = gaussian_reader('cf3_dibenzofuran.log', 'input')

## Extract connectivity data
#connectivity = gaussian_connectivity_extractor('cf3_dibenzofuran.log', total_lines)

#p(connectivity)
