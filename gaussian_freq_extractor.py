import sys
import os
sys.path.append(r''+os.getcwd()+'\FLPs\Python')
#os.chdir(r''+os.getcwd()+'\FLPs\Python')
from gaussian_reader import gaussian_reader
from utility_functions import *
import pandas as pd

def gaussian_freq_extractor(total_lines, file_name):
    positions = find_index(total_lines, ' Harmonic frequencies (cm**-1)')
    if not positions:
        return ['No freq calculations found for ' + file_name]
    
    last_pos = positions.pop()
    freq_lines = []
    n = last_pos
    while True:
        current_line = total_lines[n]
        freq_lines.append(current_line)
        if current_line.startswith(' --------'):
            break
        n += 1

    freq_pos = find_index(freq_lines, ' Frequencies -- ')
    freq_pos = [num - 2 for num in freq_pos]
    step = freq_pos[1] - freq_pos[0]

    # Initialize data storage
    columns = []
    data = []
    freq_values = []

    # Extract AN column
    AN = [int(freq_lines[n].split()[1]) for n in range(freq_pos[0] + 7, freq_pos[0] + step)]
    columns.append(('AN', ''))
    data.append(AN)

    # Extract frequency coordinates
    for m in freq_pos:
        for i in range(3):  # Three frequencies per block
            freq_label = 'Freq ' + freq_lines[m].split()[i]
            freq_value = freq_lines[m+2].split()[i+2]
            for coord in ['X', 'Y', 'Z']:
                col_label = (freq_label, coord)
                values = [
                    float(freq_lines[n].split()[2 + 3 * i + ['X', 'Y', 'Z'].index(coord)])
                    for n in range(m + 7, m + step)
                ]
                columns.append(col_label)
                data.append(values)
            freq_values.append(freq_value)

    # Build DataFrame
    index = pd.MultiIndex.from_tuples(columns, names=["Freq", "Coor"])
    freqs = pd.DataFrame({col: values for col, values in zip(index, data)})
    values_index = ['Freq ' + str(i+1) for i in range(len(freq_values))]
    freq_values = pd.DataFrame({'Values': freq_values}, index=values_index)

    freqs.index = range(1, len(freqs) + 1)
    return [freqs, freq_values]

#file_name = 'cf3_metriazole_ts1.log'
#total_lines = gaussian_reader(file_name, 'input_ts')
#x = gaussian_freq_extractor(total_lines, file_name)