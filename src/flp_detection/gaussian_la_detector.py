from utility_functions import all_same

def gaussian_la_detector(file_name, nbo):
    if isinstance(nbo, str):
        return [f'NBO missing for {file_name}']
    if not nbo:
        return [f'Invalid or empty NBO data for {file_name}']

    la_entries = []
    la_indices = []

    for entry in nbo:
        if not isinstance(entry, (list, tuple)) or len(entry) < 6:
            continue

        if str(entry[1]) != 'LP*(':
            continue

        # Detect where the atom label is
        label2 = str(entry[2]) if len(entry) > 2 else ''
        label3 = str(entry[3]) if len(entry) > 3 else ''

        is_la = any(x in label2 for x in ['B', 'Al']) or any(x in label3 for x in ['B', 'Al'])
        if not is_la:
            continue

        try:
            orb_num = int(float(entry[0]))

            # Case 1: Al-style format: ['63.', 'LP*(', '1)Al', '36', 0.60407, -0.01594]
            if any(x in label2 for x in ['B', 'Al']) and str(entry[3]).isdigit():
                atom_index = int(entry[3])
                energy = float(entry[5])

            # Case 2: B-style format: label/index shifted right
            elif any(x in label3 for x in ['B', 'Al']):
                atom_index = int(entry[4])
                energy = float(entry[6])

            else:
                continue

            la_entries.append([entry, [atom_index, orb_num]])
            la_indices.append(atom_index)

        except (ValueError, TypeError, IndexError):
            continue

    if not la_entries:
        return [f'No Lewis acid found for {file_name}']

    if len(la_entries) > 1:
        if all_same(la_indices):
            min_energy_entry = min(la_entries, key=lambda x: float(x[0][-1]))
            return min_energy_entry
        return [f'More than 1 possible Lewis acids found for {file_name}']

    return la_entries[0]