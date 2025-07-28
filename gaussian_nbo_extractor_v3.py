from itertools import takewhile
from utility_functions_v2 import *
from gaussian_reader_v2 import gaussian_reader
import os

os.chdir(r"C:\Users\francesc.penas\Dropbox\cdf francesc\Fe_porph\calcs\feIII\fetpp\quartet\opt")

def gaussian_nbo_extractor(file_name: str, total_lines: list[str]) -> list:
    """
    Extracts Natural Bond Orbital (NBO) information from a Gaussian log file.
    
    Parameters:
    - file_name (str): The name of the Gaussian log file.
    - total_lines (list of str): Lines of the Gaussian log file as strings.
    
    Returns:
    - list: A list of extracted NBO data or a message if no NBO calculations are found.
    """
    # Find positions where 'Natural Bond Orbitals (Summary):' appears in the file
    positions = find_index(total_lines, ' Natural Bond Orbitals (Summary):')
    if not positions:
        return [f'No NBO calculations found for {file_name}']

    # Start extracting from the last occurrence
    last_pos = positions.pop()
    nbo_data = []
    current_line_idx = last_pos + 6

    while current_line_idx < len(total_lines):
        current_line = total_lines[current_line_idx].strip()  # Strip whitespace

        # Check for end of the NBO section
        if current_line.startswith('       ------'):
            if total_lines[current_line_idx + 8].startswith('Molecular unit'):
                current_line_idx += 9  # Skip to the next section
                continue
            else:
                break

        # Filter lines based on leading whitespace and extract data
        leading_spaces = len(list(takewhile(str.isspace, total_lines[current_line_idx])))
        if leading_spaces <= 10:
            processed_line = parse_nbo_line(total_lines[current_line_idx][:61])
            if processed_line:
                nbo_data.append(processed_line)
        
        current_line_idx += 1

    # Return extracted NBO data or a placeholder message
    return nbo_data

def parse_nbo_line(line: str) -> list:
    """
    Parses a line of NBO data into its components.
    
    Parameters:
    - line (str): A line from the NBO summary section.
    
    Returns:
    - list: Parsed data or None if the line contains invalid data.
    """
    if '************' in line[48:60]:
        return None
    descriptor = line[:7].strip()
    elements = line[7:31].split()
    try:
        value_1 = float(line[31:48].strip())
        value_2 = float(line[48:60].strip())
    except ValueError:
        return None
    return [descriptor] + elements + [value_1, value_2]

print(os.listdir())
# Define the path to your Gaussian log file
file_name = 'fetpp_high_nbo.log'

# Read the content of the Gaussian log file
# `gaussian_reader` should load the file into a list of lines (one string per line)
total_lines = from_pkl('fetpp_high_nbo_totlines.pkl')

# Run the gaussian_nbo_extractor function with the test file content
extracted_nbo_data = gaussian_nbo_extractor(file_name, total_lines)

# Print the output to verify
#print("Extracted NBO Data:")
#for line in extracted_nbo_data:
#    print(line)