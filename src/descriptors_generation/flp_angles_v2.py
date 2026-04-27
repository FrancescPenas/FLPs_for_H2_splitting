from utility_functions import vector, angle, length
import numpy as np

def get_plane_normal(atom_indices, coordinates):
    # Given three atom indices, calculate the normal vector to the plane defined by those atoms.
    if len(atom_indices) != 3:
        return None  # Not enough bonds to form a plane
    coords = [coordinates.loc[i - 1, ['X', 'Y', 'Z']].values.astype(float) for i in atom_indices]
    return np.cross(vector(coords[0], coords[1]), vector(coords[0], coords[2]))

def choose_normal(vect, nbonds, center_coords, coordinates):
    # Choose the correct normal vector direction based on the positions of the bonded atoms.
    if len(nbonds) == 3:
        bond_coords = [coordinates.loc[i - 1, ['X', 'Y', 'Z']].values.astype(float) for i in nbonds]
        sum_1 = np.linalg.norm(bond_coords - (center_coords + vect), axis=1).sum()
        sum_2 = np.linalg.norm(bond_coords - (center_coords - vect), axis=1).sum()
        return vect if sum_1 > sum_2 else -vect
    else:
        return vect

def flp_angles(file_name, coordinates, connectivity, la_coords, lb_coords, la_eindex, lb_eindex):
    # Calculate the angles related to the FLP geometry.
    lalb_vect = vector(la_coords, lb_coords)[0]
    connec = [item for item in connectivity if len(item) == 2]
    nlabonds, nlbbonds = [], []
    
    for bond in connec:
        if la_eindex[0] in bond:
            nlabonds.append([atom for atom in bond if atom != la_eindex[0]][0])
        if lb_eindex[0] in bond:
            nlbbonds.append([atom for atom in bond if atom != lb_eindex[0]][0])
    
    la_vect = get_plane_normal(nlabonds, coordinates)

    la_vect = choose_normal(la_vect, nlabonds, la_coords, coordinates)

    if len(nlbbonds) == 3:

        lb_vect = get_plane_normal(nlbbonds, coordinates)

        lb_vect = choose_normal(lb_vect, nlbbonds, lb_coords, coordinates)
    else:
        lb_vect = None

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
    
    la_vect_u = la_vect / length(la_vect)
    lb_vect_u = lb_vect / length(lb_vect)
    lalb_vect_u = lalb_vect / length(lalb_vect)

    la_perp_vect = np.cross(la_vect_u, lalb_vect_u)
    lb_perp_vect = np.cross(lb_vect_u, (-1 * lalb_vect_u))

    dihed = np.degrees(angle(la_perp_vect, lb_perp_vect))
    
    ang_la = np.degrees(angle(la_vect_u, lalb_vect_u))
    
    ang_lb = np.degrees(angle(lb_vect_u, -lalb_vect_u))

    ang_lalb = ang_la + ang_lb
    
    direct_ang = np.degrees(angle(la_vect_u, lb_vect_u))
    
    return dihed, ang_lalb, direct_ang