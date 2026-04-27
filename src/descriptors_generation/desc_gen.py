import numpy as np
from atomic_data import a_weights
from utility_functions import vector, angle
from descriptors_generation.flp_angles_v2 import flp_angles

def extract_energies(nbo, la_eindex, lb_eindex):
    # Extract the energies for LA and LB from the NBO data using the provided indices.
    la_nbo = next(item for item in nbo if float(item[0]) == float(la_eindex[1]))
    lb_nbo = next(item for item in nbo if float(item[0]) == float(lb_eindex[1]))

    la_ener = float(la_nbo[-1])
    lb_ener = float(lb_nbo[-1])
    
    return la_ener, lb_ener

def lp_coordinates(coordinates, la_eindex, lb_eindex):
    # Extract the coordinates for LA and LB from the coordinates DataFrame using the provided indices.
    try:
        la_coords = coordinates[coordinates['Center number'] == str(la_eindex[0])].loc[:, ['X', 'Y', 'Z']].values.astype(float)
        lb_coords = coordinates[coordinates['Center number'] == str(lb_eindex[0])].loc[:, ['X', 'Y', 'Z']].values.astype(float)
        return la_coords, lb_coords
    except KeyError:
        raise ValueError(f"Atom indices {la_eindex[2]} or {lb_eindex[2]} not found in coordinates DataFrame.")

def calculate_distances(la_coords, lb_coords):
    # Calculate the distance between LA and LB.
    dist_la_lb = np.linalg.norm(la_coords - lb_coords)
        
    # Calculate the midpoint between LA and LB.
    mid_point = (la_coords + lb_coords) / 2
    mid_point = mid_point.tolist()[0]
    return dist_la_lb, mid_point

def calculate_molec_weight(coordinates: pd.DataFrame) -> float:
    # Calculate the molecular weight of the molecule based on the atomic numbers and their corresponding weights.
    atomic_numbers = coordinates["Atomic Number"].astype(int) - 1  # Adjust for 0-based indexing
    return np.sum([a_weights[i] for i in atomic_numbers])

def calculate_elecfield(charge, mid_dist):
    # Calculate the electric field at the midpoint between LA and LB using the formula E = q / r^2, where q is the charge and r is the distance from the charge to the point of interest.
    electric_field = charge / (mid_dist**2)
    return electric_field

def calculate_elecpot(la_charge, lb_charge, mid_dist):
    # Calculate the electrostatic potential at the midpoint between LA and LB using the formula V = k * (q_la * q_lb) / r, where k is Coulomb's constant (which we can set to 1 for simplicity), q_la and a_lb are the charges of LA and LB respectively, and r is the midpoint distance between LA and LB.
    electric_pot = (la_charge * lb_charge) / mid_dist
    return electric_pot

def desc_gen(file_name, dataextracted, dataflp, esp_charges_dir=False):
    # Generate descriptors for a given molecule based on the extracted data and FLP information.
    dot_index = file_name.find('.')
    coordinates = dataextracted[2][file_name]
    nbo = dataextracted[3][file_name]
    connectivity = dataextracted[4][file_name]
    npa = dataextracted[5][file_name]
    if esp_charges_dir:
        esp = dataextracted[6][file_name[:dot_index] + '_chelpg.log']
    else:
        esp = dataextracted[6][file_name]
    la_eindex = dataflp[0][file_name]
    lb_eindex = dataflp[1][file_name]
    print(file_name)

    # Extract energies for LA and LB orbitals
    la_ener, lb_ener = extract_energies(nbo, la_eindex, lb_eindex)
    
    # Extract coordinates for LA and LB
    la_coords, lb_coords = lp_coordinates(coordinates, la_eindex, lb_eindex)

    # Calculate distance between LA and LB and the midpoint
    dist_la_lb, mid_point = calculate_distances(la_coords, lb_coords)
    
    # Calculate molecular weight
    molec_weight = calculate_molec_weight(coordinates)

    # Calculate angles related to the FLP geometry
    dihed, ang_lalb, direct_ang = flp_angles(file_name, coordinates, connectivity, la_coords, lb_coords, la_eindex, lb_eindex)

    # Extract charges for LA and LB from NPA and ESP data
    la_nat_charge = npa[npa['No'] == str(la_eindex[0])].loc[:, ['Natural Charge']].values.astype(float).tolist()[0][0]
    lb_nat_charge = npa[npa['No'] == str(lb_eindex[0])].loc[:, ['Natural Charge']].values.astype(float).tolist()[0][0]
    la_esp_charge = esp[esp['Center number'] == str(la_eindex[0])].loc[:, ['ESP charge']].values.astype(float).tolist()[0][0]
    lb_esp_charge = esp[esp['Center number'] == str(lb_eindex[0])].loc[:, ['ESP charge']].values.astype(float).tolist()[0][0]

    # Calculate electric fields at the midpoint for both LA and LB using both NPA and ESP charges
    mid_dist = np.linalg.norm(mid_point - la_coords)  # Distance from LA to midpoint (same as distance from LB to midpoint)

    la_npa_elec_field = calculate_elecfield(la_nat_charge, mid_dist)
    lb_npa_elec_field = calculate_elecfield(lb_nat_charge, mid_dist)
    la_esp_elec_field = calculate_elecfield(la_esp_charge, mid_dist)
    lb_esp_elec_field = calculate_elecfield(lb_esp_charge, mid_dist)

    # Calculate electrostatic potential at the midpoint using both NPA and ESP charges
    npa_elec_pot = calculate_elecpot(la_nat_charge, lb_nat_charge, mid_dist)
    esp_elec_pot = calculate_elecpot(la_esp_charge, lb_esp_charge, mid_dist)
    
    return la_ener, lb_ener, dist_la_lb, molec_weight, dihed, ang_lalb, direct_ang, la_nat_charge, lb_nat_charge, la_esp_charge, lb_esp_charge, la_npa_elec_field, lb_npa_elec_field, la_esp_elec_field, lb_esp_elec_field, npa_elec_pot, esp_elec_pot