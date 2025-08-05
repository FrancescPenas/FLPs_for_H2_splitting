import sys
import os
from tkinter import Label
sys.path.append(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
os.chdir(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from utility_functions import *
from atomic_data import a_weights

label_dict = {
    'f_pyrazole.log': 1,
    'h_indole.log': 2,
    'me_indole.log': 3,
    'cl_indole.log': 4,
    'tbu_indole.log': 5,
    'cl_nmeimidazole.log': 6,
    'f_nmeimidazole.log': 7,
    'h_meimidazolol.log': 8,
    'tbu_nmeimidazole.log': 9,
    'h_meimidazolamine.log': 16,
    'h_dimethylimidazole.log': 10,
    'h_nmeimidazole.log': 11,
    'no2_nmeimidazole.log': 13,
    'h_menitroimidazole.log': 14,
    'h_fnmeimidazole.log': 12,
    'h_meimidazolecarboxylic.log': 15,
    'me_metriazole.log': 17,
    'ethy_metriazole.log': 18,
    'cl_metriazole.log': 19,
    'f_metriazole.log': 20,
    'f_dimetriazole.log': 21,
    'ipr_metriazole.log': 22,
    'f_clmetriazole.log': 23,
    'ph_metriazole.log': 24,
    'f_cnmetriazole.log': 25,
    'tbu_metriazole.log': 26,
    'f_fmetriazole.log': 27,
    'mesityl_metriazole.log': 28,
    'f_metriazolamine.log': 29,
    'ccl3_metriazole.log': 30,
    'cn_metriazole.log': 31,
    'cyclo_metriazole.log': 32,
    'perfluorophenyl_metriazole.log': 33,
    'oh_metriazole.log': 34,
    '4meborinane_metriazole.log': 35,
    'cooh_metriazole.log': 36,
    'cf3_metriazole.log': 37,
    'h_metriazole.log': 38,
    'bicyclo_metriazole.log': 39,
    'no2_metriazole.log': 40,
    'borole_dimethylaniline.log': 41,
    'h_dimethylaniline.log': 42,
    'me_dimethylaniline.log': 43,
    'ethy_dimethylaniline.log': 44,
    'me_aniline.log': 45,
    'cl_dimethylaniline.log': 46,
    'me_tetramethylbenzopiperidine.log': 47,
    'f_dimethylaniline.log': 48,
    'f_tetramethylbenzopiperidine.log': 49,
    'ipr_dimethylaniline.out': 50,
    'ph_dimethylaniline.log': 51,
    '4meborinane_dimethylaniline.log': 52,
    'tbu_dimethylaniline.log': 53,
    'mesityl_dimethylaniline.log': 54,
    'cyclo_dimethylaniline.log': 55,
    'bicyclo_dimethylaniline.log': 56,
    'cyclo5_dimethylaniline.log': 57,
    'h_quinoline.log': 60,
    'me_benzopyrrolidine.log': 58,
    'f_quinoline.log': 61,
    'me_cinnoline.log': 59,
    'me_quinoline.log': 62,
    'cl_quinoline.log': 63,
    'tbu_quinoline.log': 64,
    'no2_quinoline.log': 65,
    'me_phquinolinamine.out': 66,
    'me_mequinolinamine.out': 67,
    'h_benzopiperidine.log': 68,
    'cl_benzopiperidine.log': 69,
    'f_benzopiperidine.log': 70,
    'me_benzopiperidine.log': 71,
    'tbu_benzopiperidine.log': 72,
    'cf3_benzopiperidine.log': 73,
    'no2_benzopiperidine.log': 74,
    'h_naphthalene.log': 75,
    'me_naphthalene.log': 76,
    'cl_naphthalene.log': 77,
    'tbu_naphthalene.log': 78,
    'me_indenoPy.log': 79,
    'tbu_indenoPy.log': 80,
    'f_indenoPy.log': 81,
    'h_biphenylene.log': 82,
    'tbu_biphenylene.log': 83,
    'me_biphenylene.log': 84,
    'cyclo_biphenylene.log': 85,
    'cl_biphenylene.log': 86,
    'f_biphenylene.log': 87,
    'bicyclo_biphenylene.log': 88,
    'ph_biphenylene.log': 89,
    'h_xanthene.log': 90,
    'me_xanthene.log': 91,
    'cyclo_xanthene.log': 92,
    'cl_xanthene.log': 93,
    'tbu_xanthene.log': 94,
    'bicyclo_xanthene.log': 95,
    'ph_xanthene.log': 96,
    'me_naphtoxanthene.log': 97,
    'cl_naphtoxanthene.log': 98,
    'ph_naphtoxanthene.log': 99,
    'tbu_naphtoxanthene.log': 100,
    'bicyclo_naphtoxanthene.log': 101,
    'me_Bbenzoxanthene.log': 102,
    'me_tribenzooxepine.log': 103,
    'h_dibenzofuran.log': 104,
    'f_dibenzofuran.log': 105,
    'me_dibenzofuran.log': 106,
    'ph_dibenzofuran.log': 107,
    'cl_dibenzofuran.log': 108,
    'tbu_dibenzofuran.log': 109,
    'cf3_dibenzofuran.log': 110,
    'cyclo_dibenzofuran.log': 111,
    'bicyclo_dibenzofuran.log': 112
}

def calculate_energies(nbo, la_eindex, lb_eindex):
    """Calculate energies and related descriptors."""
    la_nbo = next(item for item in nbo if float(item[0]) == float(la_eindex[1]))
    lb_nbo = next(item for item in nbo if float(item[0]) == float(lb_eindex[1]))
    
    la_ener = float(la_nbo[6])
    lb_ener = float(lb_nbo[7])
    
    return la_ener, lb_ener

def lp_coordinates(coordinates, la_eindex, lb_eindex):
    try:
        la_coords = coordinates[coordinates['Center number'] == str(la_eindex[0])].loc[:, ['X', 'Y', 'Z']].values.astype(float)
        lb_coords = coordinates[coordinates['Center number'] == str(lb_eindex[0])].loc[:, ['X', 'Y', 'Z']].values.astype(float)
        return la_coords, lb_coords
    except KeyError:
        raise ValueError(f"Atom indices {la_eindex[2]} or {lb_eindex[2]} not found in coordinates DataFrame.")

def calculate_distances(la_coords, lb_coords):
    # Compute the distance
    dist_la_lb = np.linalg.norm(la_coords - lb_coords)
        
    # Calculate the midpoint
    mid_point = (la_coords + lb_coords) / 2
    mid_point = mid_point.tolist()[0]
    return dist_la_lb, mid_point

def calculate_molec_weight(coordinates: pd.DataFrame) -> float:
    """Compute the molecular weight from a pandas DataFrame of atomic coordinates."""
    atomic_numbers = coordinates["Atomic Number"].astype(int) - 1  # Adjust for 0-based indexing
    return np.sum([a_weights[i] for i in atomic_numbers])

def flp_angles(file_name, coordinates, connectivity, la_coords, lb_coords, la_eindex, lb_eindex):
    lalb_vect = vector(la_coords, lb_coords)[0]
    connec = [item for item in connectivity if len(item) == 2]
    nlabonds, nlbbonds = [], []
    
    for bond in connec:
        if la_eindex[0] in bond:
            nlabonds.append([atom for atom in bond if atom != la_eindex[0]][0])
        if lb_eindex[0] in bond:
            nlbbonds.append([atom for atom in bond if atom != lb_eindex[0]][0])
    
    def get_plane_normal(atom_indices):
        if len(atom_indices) != 3:
            return None  # Not enough bonds to form a plane
        coords = [coordinates.loc[i - 1, ['X', 'Y', 'Z']].values.astype(float) for i in atom_indices]
        return np.cross(vector(coords[0], coords[1]), vector(coords[0], coords[2]))
    
    la_vect = get_plane_normal(nlabonds)
    lb_vect = get_plane_normal(nlbbonds) if len(nlbbonds) == 3 else None
    
    if la_vect is None:
        print(f'LA underbonded for {file_name}')
        return None
    if lb_vect is None:
        if len(nlbbonds) == 2:
            slb1 = coordinates.loc[nlbbonds[0] - 1, ['X', 'Y', 'Z']].values.astype(float)
            slb2 = coordinates.loc[nlbbonds[1] - 1, ['X', 'Y', 'Z']].values.astype(float)
            p = (slb1 + slb2) / 2
            lb_vect = vector(p, lb_coords)[0]
        else:
            print(f'LB underbonded for {file_name}')
            return None
    
    la_vect_u = (la_vect / np.linalg.norm(la_vect))
    lb_vect_u = (lb_vect / np.linalg.norm(lb_vect))
    
    la_perp_vect = np.cross(la_vect, lalb_vect)
    lb_perp_vect = np.cross(lb_vect, (-1 * lalb_vect))
    
    dihed = np.degrees(angle(la_perp_vect, lb_perp_vect))
    dihed = 180 - dihed if dihed > 90 else dihed
    
    ang_la = np.degrees(angle(la_vect_u, lalb_vect))
    ang_la = 180 - ang_la if ang_la > 90 else ang_la
    
    ang_lb = np.degrees(angle(lb_vect_u, -lalb_vect))
    ang_lb = 180 - ang_lb if ang_lb > 90 else ang_lb

    ang_lalb = ang_la + ang_lb
    
    direct_ang = np.degrees(angle(la_vect, lb_vect))
    
    return dihed, ang_lalb, direct_ang

def calculate_elecfield(charge, mid_dist):
    electric_field = charge / (mid_dist**2)
    return electric_field

def calculate_elecpot(la_charge, lb_charge, mid_dist):
    electric_pot = (la_charge * lb_charge) / mid_dist
    return electric_pot

def process_file(file_name, dataextracted, dataflp):
    dot_index = file_name.find('.')
    """Process a single file to compute all descriptors."""
    coordinates = dataextracted[2][file_name]
    nbo = dataextracted[3][file_name]
    connectivity = dataextracted[4][file_name]
    npa = dataextracted[5][file_name]
    esp = dataextracted[6][file_name[:dot_index] + '_chelpg.log']
    la_eindex = dataflp[0][file_name]
    lb_eindex = dataflp[1][file_name]
    print(file_name)
    # Calculate energies
    la_ener, lb_ener = calculate_energies(nbo, la_eindex, lb_eindex)
    
    # Calculate lp coordinates
    la_coords, lb_coords = lp_coordinates(coordinates, la_eindex, lb_eindex)

    # Calculate distances
    dist_la_lb, mid_point = calculate_distances(la_coords, lb_coords)
    
    # Calculate molec_weight
    molec_weight = calculate_molec_weight(coordinates)

    # FLP angles
    dihed, ang_lalb, direct_ang = flp_angles(file_name, coordinates, connectivity, la_coords, lb_coords, la_eindex, lb_eindex)

    # Calculate charges
    la_nat_charge = npa[npa['No'] == str(la_eindex[0])].loc[:, ['Natural Charge']].values.astype(float).tolist()[0][0]
    lb_nat_charge = npa[npa['No'] == str(lb_eindex[0])].loc[:, ['Natural Charge']].values.astype(float).tolist()[0][0]
    la_esp_charge = esp[esp['Center number'] == str(la_eindex[0])].loc[:, ['ESP charge']].values.astype(float).tolist()[0][0]
    lb_esp_charge = esp[esp['Center number'] == str(lb_eindex[0])].loc[:, ['ESP charge']].values.astype(float).tolist()[0][0]

    # Calculate eletrostatic field
    mid_dist = np.linalg.norm(mid_point)
    la_npa_elec_field = calculate_elecfield(la_nat_charge, mid_dist)
    lb_npa_elec_field = calculate_elecfield(lb_nat_charge, mid_dist)
    la_esp_elec_field = calculate_elecfield(la_esp_charge, mid_dist)
    lb_esp_elec_field = calculate_elecfield(lb_esp_charge, mid_dist)

    # Calculate eletrostatic potential
    npa_elec_pot = calculate_elecpot(la_nat_charge, lb_nat_charge, mid_dist)
    esp_elec_pot = calculate_elecpot(la_esp_charge, lb_esp_charge, mid_dist)
    
    # Extrac energies from csv files
    #free_ener = 

    return la_ener, lb_ener, dist_la_lb, molec_weight, dihed, ang_lalb, direct_ang, la_nat_charge, lb_nat_charge, la_esp_charge, lb_esp_charge, la_npa_elec_field, lb_npa_elec_field, la_esp_elec_field, lb_esp_elec_field, npa_elec_pot, esp_elec_pot

def desc_gen(pkl_dataextracted, pkl_dataflp, free_ener_csv, free_ener_barr_csv, feha_ener_csv, fepa_ener_csv, pkl_outfile='data_out'):
    """Main function to generate descriptors for all files."""
    dataextracted, dataflp = from_pkl(pkl_dataextracted), from_pkl(pkl_dataflp)
    file_name_list = dataextracted[1]

    data = [process_file(file_name, dataextracted, dataflp) for file_name in file_name_list]

    # FEHA
    feha_ener_df = pd.read_csv(feha_ener_csv)
    feha_ener_dict = feha_ener_df.to_dict(orient="list")
    feha_ener_list = [feha_ener_dict[file_name][0] for file_name in file_name_list]

    # FEPA
    fepa_ener_df = pd.read_csv(fepa_ener_csv)
    fepa_ener_dict = fepa_ener_df.to_dict(orient="list")
    fepa_ener_list = [fepa_ener_dict[file_name][0] for file_name in file_name_list]

    # Free energies
    free_ener_df = pd.read_csv(free_ener_csv)
    free_ener_dict = free_ener_df.to_dict(orient="list")
    free_ener_list = [free_ener_dict[file_name][0] for file_name in file_name_list]

    # Free energy barriers
    free_ener_barr_df = pd.read_csv(free_ener_barr_csv)
    free_ener_barr_dict = free_ener_barr_df.to_dict(orient="list")
    free_ener_barr_list = [free_ener_barr_dict[file_name][0] for file_name in file_name_list]

    # Labels
    labels_list = [label_dict[file_name] for file_name in file_name_list]

    data = [(a,) + (b,) + (c,) + (d,) + (e,) + (f,) + sublist for a, b, c, d, e, f, sublist in zip(labels_list, file_name_list, free_ener_list, free_ener_barr_list, feha_ener_list, fepa_ener_list, data)]

    # Convert results to DataFrame and save
    df = pd.DataFrame(data)
    df.columns = ['labels', 'file_name', 'free_ener_reac', 'free_ener_barr', 'FEHA', 'FEPA', 'E_p(LA)', 'E_p(LB)', 'd', 'Mw', 'γ', 'λ', 'Φ', 'q_la(NPA)', 'q_lb(NPA)', 'q_la(ESP)', 'q_lb(ESP)', 'EF_la(NPA)', 'EF_lb(NPA)', 'EF_la(ESP)', 'EF_lb(ESP)', 'EP(NPA)', 'EP(ESP)']
    to_pkl(df, pkl_outfile + '.pkl')
    df.to_csv(pkl_outfile + '.csv', index=False)
    
    return df

#x = from_pkl('input_data.pkl')
#y = from_pkl('data_flp.pkl')
z = desc_gen('input_data.pkl', 'data_flp.pkl', 'h_split_ener.csv', 'h_split_ener_barr.csv', 'feha_energies.csv', 'fepa_energies.csv', pkl_outfile='final_data')