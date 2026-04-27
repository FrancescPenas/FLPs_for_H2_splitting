import os

def gaussian_reader(file_name, input_dir='.'):
    if input_dir == '.':
        dir_path = os.getcwd()
    else:
        dir_path = os.path.abspath(input_dir)
    file_path = os.path.join(dir_path, file_name)  # Cross-platform path joining
    with open(file_path, "r") as file_read:
        total_lines = file_read.readlines()
    return total_lines