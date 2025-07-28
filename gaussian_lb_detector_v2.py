from utility_functions_v2 import *
from gaussian_la_detector_v2 import gaussian_la_detector
from dist_calc_v2 import dist_calc

def gaussian_lb_detector(file_name, coordinates, nbo, connectivity, la):
    
    # Initialize variables
    lb_candidates = []
    bn_distances = []

    # Check if 'la' is a valid index or integer, else return error message
    if isinstance(la, str):
        return ['No Lewis acid found for ' + file_name]

    # Process each lone pair in the NBO list
    for orbital in nbo:
        if orbital[1] == 'LP' and orbital[4] == 'N':
            # Check if the Lewis acid and this nitrogen atom are directly connected
            B = int(la)
            N = int(orbital[5])
            bond = sorted([B, N])
            if bond not in connectivity:
                # Calculate distance if no direct connection exists
                bn_distance = dist_calc(coordinates, B, N)
                bn_distances.append(bn_distance)
                lb_candidates.append([orbital, [int(orbital[5]), int(float(orbital[0]))], bn_distance])

    # Select the closest Lewis base, if any are found
    if len(lb_candidates) == 1:
        return lb_candidates
    elif lb_candidates:
        min_dist_index = bn_distances.index(min(bn_distances))
        return [lb_candidates[min_dist_index]]
    else:
        return ['No Lewis base found for ' + file_name]

#file_name = 'bicyclo_metriazole.log'
#x = from_pkl('input_data.pkl')
#coordinates = x[2][file_name]
#nbo = x[3][file_name]
#connectivity = x[4][file_name]
#la = gaussian_la_detector(file_name, nbo)
#y = gaussian_lb_detector(file_name, coordinates, nbo, connectivity, la[1][0])