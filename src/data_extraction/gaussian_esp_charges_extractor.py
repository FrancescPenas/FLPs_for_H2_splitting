from utility_functions import find_index
import pandas as pd

def gaussian_esp_charges_extractor(file_name, total_lines):
    positions = find_index(total_lines, ' ESP charges:')
    if positions == []:
        return 'No ESP charges found for'+file_name
    else:
        last_pos = positions.pop()
        esp_charges = []
        n = last_pos + 2
        while True:
            current_line = total_lines[n]
            if ' Sum of ESP' in current_line:
                break
            current_line = current_line.split()
            esp_charges += [[current_line[0], current_line[1], float(current_line[2])]]
            n += 1

         # Create DataFrame
        columns = ['Center number', 'Atom', 'ESP charge']
        esp_char_df = pd.DataFrame(esp_charges, columns=columns)

        return esp_char_df