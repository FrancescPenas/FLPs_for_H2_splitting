import sys
import os
sys.path.append(r''+os.getcwd()+'\FLPs\Python\code')
#os.chdir(r''+os.getcwd()+'\FLPs\Python\code')
os.chdir(r"C:\Users\francesc.penas\Dropbox\cdf francesc\Fe_porph\calcs\feIII\fetpp\quartet\opt")
from utility_functions import *

def gaussian_reader(file_name, input_dir='.'):
    if input_dir == '.':
        dir_path = os.getcwd()
    else:
        dir_path = os.path.abspath(input_dir)
    file_path = os.path.join(dir_path, file_name)  # Cross-platform path joining
    with open(file_path, "r") as file_read:
        total_lines = file_read.readlines()
    return total_lines

# Usage
total_lines = gaussian_reader('fetpp_high_nbo.log')
to_pkl(total_lines, 'fetpp_high_nbo_totlines.pkl')