import pandas as pd
from relabeling import label_dict, family_dict, external_label_dict, external_family_dict
from utility_functions import to_pkl

def dataset_builder(data, descriptors, free_ener_csv, free_ener_barr_csv, feha_ener_csv, fepa_ener_csv, pkl_outfile='dataset', mode='train'):
    # Builds a dataset for machine learning by combining labels, file names, energies, and descriptors for each molecule.
    file_name_list = data[1]

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
    if mode == 'train':
        labels_list = [label_dict[file_name] for file_name in file_name_list]
        # Map labels to families
        label_to_family = {
            label: family
            for family, r in family_dict.items()
            for label in r
        }
    elif mode == 'test':
        labels_list = [external_label_dict[file_name] for file_name in file_name_list]
        # Map labels to families
        label_to_family = {
            label: family
            for family, r in external_family_dict.items()
            for label in r
        }

    family_list = [label_to_family[label] for label in labels_list]

    # Combine all data into a list of tuples for DataFrame creation
    data = [(a,) + (b,) + (c,) + (d,) + (e,) + (f,) + (g,) + sublist for a, b, c, d, e, f, g, sublist in zip(labels_list, family_list, file_name_list, free_ener_list, free_ener_barr_list, feha_ener_list, fepa_ener_list, descriptors)]

    # Create a DataFrame and save it as a pickle and CSV file.
    df = pd.DataFrame(data)
    df.columns = ['labels', 'family', 'file_name', 'free_ener_reac', 'free_ener_barr', 'FEHA', 'FEPA', 'E$_{p}$(LA)', 'E$_{p}$(LB)', 'd', 'Mw', 'γ', 'λ', 'Φ', 'q$_{la}$(NPA)', 'q$_{lb}$(NPA)', 'q$_{la}$(ESP)', 'q$_{lb}$(ESP)', 'EF$_{la}$(NPA)', 'EF$_{lb}$(NPA)', 'EF$_{la}$(ESP)', 'EF$_{lb}$(ESP)', 'EP(NPA)', 'EP(ESP)']
    to_pkl(df, pkl_outfile + '.pkl')
    df.to_csv(pkl_outfile + '.csv', index=False)
    
    return df