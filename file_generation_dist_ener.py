import sys
import os
sys.path.append(r''+os.getcwd()+'\FLPs\Python\code')
os.chdir(r''+os.getcwd()+'\FLPs\Python\code')
from default_functions import *
from atomic_data import symbols
import random
from gaussian_reader import gaussian_reader
from gaussian_freq_extractor import gaussian_freq_extractor

def h2_ts_detector(file_name, coor_ts, freqs):
    freq_coor, freq_values = freqs
    freq_values = freq_values.astype(float).squeeze().tolist()

    num_negatives = sum([1 for num in freq_values if num < 0])
    if num_negatives == 0:
        print(f'No negative frequencies for {file_name}')
        return
    elif num_negatives > 1:
        print(f'More than one negative frequency for {file_name}')
        return

    # Identify significant modes
    im_freq = freq_coor[['AN', 'Freq 1']].copy()
    im_freq.columns = ['AN', 'X', 'Y', 'Z']
    im_freq['Module'] = np.sqrt(im_freq['X']**2 + im_freq['Y']**2 + im_freq['Z']**2)
    df_an_1 = im_freq[im_freq['AN'] == 1]
    top_two_modules = df_an_1.sort_values(by='Module', ascending=False).head(2)
    ts_h_labels = top_two_modules.index.tolist()
    return ts_h_labels

def pdb_gen(file_name, coor, symbols, residue_name="RES", chain_id="A", save_dir="."):
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the full file path
    file_path = os.path.join(save_dir, file_name + ".pdb")
    
    # Writing to the PDB file with proper formatting
    with open(file_path, "w") as f:
        atom_id = 1  # Starting atom ID
        residue_id = 1  # Starting residue ID (adjust if necessary)
        
        for row in coor:
            atom_label = int(row[0])
            atom_number = int(row[1])  # Atomic number (second column)
            atom_type = symbols[atom_number - 1]  # Get the element symbol from the list (adjusted for 0-indexing)
            
            # Ensure atom type fits within the 4 character space (right-aligned)
            atom_name = atom_type[:4].ljust(4)  # Ensure atom name is 4 characters long
            
            # Coordinates should be written with a fixed width (8.3f for x, y, z)
            x, y, z = float(row[3]), float(row[4]), float(row[5])
            
            # PDB format for an atom entry respecting spaces
            f.write(f"ATOM  {atom_label:5d} {atom_name:<4} {residue_name:>3} {chain_id:>1}{residue_id:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 25.00           {atom_type:>2}\n")
            
            atom_id += 1  # Increment atom ID for the next atom
        
        print(f"PDB file generated for {file_name} in {save_dir} directory.")

def gjf_gen(file_name, coor, symbols, charge=0, multiplicity=1, save_dir="."):
    """
    Create a .gjf file for Gaussian from a list of coordinates and atomic symbols.

    Parameters:
    - file_name: The name of the .gjf file to be generated.
    - coor: List of coordinates where each row is [atom_label, atom_number, x, y, z].
    - symbols: List of element symbols corresponding to atomic numbers.
    - molecule_name: The name of the molecule (default "Molecule").
    - charge: The molecular charge (default 0).
    - multiplicity: The spin multiplicity (default 1).
    - save_dir: Directory to save the .gjf file (default is current directory).
    """
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Construct the full file path
    file_path = os.path.join(save_dir, file_name + "_sp.gjf")
    
    # Writing to the GJF file with proper formatting
    with open(file_path, "w") as f:
        f.write(f"%nprocshared=128\n")
        f.write(f"%mem=30GB\n")
        f.write(f"%chk={file_name}.chk\n")
        f.write(f"#p freq wb97xd/genecp scrf=(solvent=toluene,pcm) nosymm pop=(regular,orbitals=20,nbo,chelpg) optcyc=300 scfcyc=300\n\n")  # Gaussian computational method and basis set
        f.write(f"{file_name}\n")
        f.write("\n")
        f.write(f"{charge} {multiplicity}\n")
        atom_types = []
        for row in coor:
            atom_label = int(row[0])
            atom_number = int(row[1])  # Atomic number
            atom_type = symbols[atom_number - 1]  # Get element symbol from the list (adjusted for 0-indexing)
            if atom_type not in atom_types:
                atom_types += [atom_type]
            # Coordinates should be written with a fixed width (8.3f for x, y, z)
            x, y, z = float(row[3]), float(row[4]), float(row[5])
            
            # Write the atom and its coordinates to the file
            f.write(f"{atom_type:>2}{x:29.8f}{y:14.8f}{z:14.8f}\n")
        
        f.write("\n")
        f.write(" ".join(atom_types)+' 0\n')
        f.write("6-311++g(d,p)\n")
        f.write("****\n")
        f.write("\n")
        f.write("\n")
    
    print(f"GJF file generated for {file_name} in {save_dir} directory.")

#random_files = random.sample(data_ts[1], 10)

file_list = ['h_quinoline_ts1',
             'h_benzopiperidine_ts1',
             'h_dimethylaniline_ts1',
             'no2_quinoline_ts1',
             '4meborinane_dimethylaniline_ts1',
             'h_dibenzofuran_ts1',
             'cl_dibenzofuran_ts1',
             'cyclo_dibenzofuran_ts1',
             'f_dibenzofuran_ts1',
             'me_dibenzofuran_ts1'
             ]

data = from_pkl('input.pkl')
data_ts = from_pkl('input_ts.pkl')

for file_name in file_list:
    coor = from_pkl('input.pkl')[2][file_name.replace('_ts1', '')+'.log']
    coor_pdb = pdb_gen(file_name.replace('_ts1', ''), coor, symbols, save_dir=(os.getcwd()+'\\ts_geoms\\'))
    coor_ts = from_pkl('input_ts.pkl')[2][file_name+'.log']
    #coor_ts_pdb = pdb_gen(file_name, coor_ts, symbols, save_dir=(os.getcwd()+'\\ts_geoms\\'))
    total_lines = gaussian_reader(file_name+'.log', 'input_ts')
    freqs = gaussian_freq_extractor(total_lines, file_name+'.log')
    ts_h_labels = h2_ts_detector(file_name, coor_ts, freqs)
    coor_ts_no_H = [item for item in coor_ts if int(item[0]) not in ts_h_labels]
    #coor_ts_no_H_pdb = pdb_gen(file_name+'_no_H', coor_ts_no_H, symbols, save_dir=(os.getcwd()+'\\ts_geoms\\'))
    coor_ts_no_H_gjf = gjf_gen(file_name, coor_ts_no_H, symbols, save_dir=(os.getcwd()+'\\ts_geoms\\'))