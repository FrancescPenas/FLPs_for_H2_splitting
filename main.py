import sys
import os

sys.path.append(os.path.abspath("src"))

from data_extraction.data_extractor import data_extractor
from flp_detection.flp_detector import flp_detector
from utility_functions import from_pkl, to_pkl, pf
from descriptors_generation.desc_gen import desc_gen
from data_preparation import dataset_builder
from data_expander import data_expander
from data_analysis.k_fold_opt import k_fold_optimization
from data_analysis.pls_analysis import pls_analysis
from data_analysis.informative_var_filter_PLS_iter import informative_variable_filter_pls
from data_analysis.informative_var_filter_PLS_iter_v2 import informative_variable_filter_pls as informative_variable_filter_pls_v2
from src.data_validation.model_validation import model_validation

#################
## Data Mining ##
#################

# First step: Extract data from the input files.

#data = data_extractor('data/input', esp_charges_dir='data/esp_charges')
#data = from_pkl('data/input_data.pkl')

# Second step: Detect FLPs.

#flp_info = flp_detector('data/input_data', 'data/flp_info')
#flp_info = from_pkl('data/flp_info.pkl')

# Third step: Generate descriptors for each molecule and save them to a pickle file for later use.

#descriptors = [desc_gen(file_name, data, flp_info, esp_charges_dir=True) for file_name in data[1]]
#to_pkl(descriptors, 'data/descriptors.pkl')
#descriptors = from_pkl('data/descriptors.pkl')

# Fifth step: Prepare the dataset for ML including labels and external information.

#dataset = dataset_builder(data, descriptors, 'data/reac_free_energies.csv', 'data/barr_free_energies.csv', 'data/feha_energies.csv', 'data/fepa_energies.csv', pkl_outfile='data/dataset', mode='train')
dataset = from_pkl('data/dataset.pkl')

## Data mining of the external validation set

# intra_B_N library

#intra_B_N_data = data_extractor('data/intra_B_N', esp_charges_dir='inside')
#intra_B_N_data = from_pkl('data/intra_B_N_data.pkl')
#intra_B_N_flp_info = flp_detector('data/intra_B_N_data', 'data/intra_B_N_flp_info')
#intra_B_N_flp_info = from_pkl('data/intra_B_N_flp_info.pkl')
#intra_B_N_descriptors = [desc_gen(file_name, intra_B_N_data, intra_B_N_flp_info, esp_charges_dir=False) for file_name in intra_B_N_data[1]]
#to_pkl(intra_B_N_descriptors, 'data/intra_B_N_descriptors.pkl')
#intra_B_N_descriptors = from_pkl('data/intra_B_N_descriptors.pkl')
#intra_B_N_dataset = dataset_builder(intra_B_N_data, intra_B_N_descriptors, 'data/intra_B_N_reac_free_energies.csv', 'data/intra_B_N_barr_free_energies.csv', 'data/intra_B_N_feha_energies.csv', 'data/intra_B_N_fepa_energies.csv', pkl_outfile='data/intra_B_N_dataset', mode='test')
intra_B_N_dataset = from_pkl('data/intra_B_N_dataset.pkl')

# inter_B_N library

#inter_B_N_data = data_extractor('data/inter_B_N', esp_charges_dir='inside')
#inter_B_N_data = from_pkl('data/inter_B_N_data.pkl')
#inter_B_N_flp_info = flp_detector('data/inter_B_N_data', 'data/inter_B_N_flp_info')
#inter_B_N_flp_info = from_pkl('data/inter_B_N_flp_info.pkl')
#inter_B_N_descriptors = [desc_gen(file_name, inter_B_N_data, inter_B_N_flp_info, esp_charges_dir=False) for file_name in inter_B_N_data[1]]
#to_pkl(inter_B_N_descriptors, 'data/inter_B_N_descriptors.pkl')
#inter_B_N_descriptors = from_pkl('data/inter_B_N_descriptors.pkl')
#inter_B_N_dataset = dataset_builder(inter_B_N_data, inter_B_N_descriptors, 'data/inter_B_N_reac_free_energies.csv', 'data/inter_B_N_barr_free_energies.csv', 'data/inter_B_N_feha_energies.csv', 'data/inter_B_N_fepa_energies.csv', pkl_outfile='data/inter_B_N_dataset', mode='test')
inter_B_N_dataset = from_pkl('data/inter_B_N_dataset.pkl')

# intra_B_P library

#intra_B_P_data = data_extractor('data/intra_B_P', esp_charges_dir='inside')
#intra_B_P_data = from_pkl('data/intra_B_P_data.pkl')
#intra_B_P_flp_info = flp_detector('data/intra_B_P_data', 'data/intra_B_P_flp_info')
#intra_B_P_flp_info = from_pkl('data/intra_B_P_flp_info.pkl')
#intra_B_P_descriptors = [desc_gen(file_name, intra_B_P_data, intra_B_P_flp_info, esp_charges_dir=False) for file_name in intra_B_P_data[1]]
#to_pkl(intra_B_P_descriptors, 'data/intra_B_P_descriptors.pkl')
#intra_B_P_descriptors = from_pkl('data/intra_B_P_descriptors.pkl')
#intra_B_P_dataset = dataset_builder(intra_B_P_data, intra_B_P_descriptors, 'data/intra_B_P_reac_free_energies.csv', 'data/intra_B_P_barr_free_energies.csv', 'data/intra_B_P_feha_energies.csv', 'data/intra_B_P_fepa_energies.csv', pkl_outfile='data/intra_B_P_dataset', mode='test')
intra_B_P_dataset = from_pkl('data/intra_B_P_dataset.pkl')

# intra_Al_N library

#intra_Al_N_data = data_extractor('data/intra_Al_N', esp_charges_dir='inside')
#intra_Al_N_data = from_pkl('data/intra_Al_N_data.pkl')
#intra_Al_N_flp_info = flp_detector('data/intra_Al_N_data', 'data/intra_Al_N_flp_info')
#intra_Al_N_flp_info = from_pkl('data/intra_Al_N_flp_info.pkl')
#intra_Al_N_descriptors = [desc_gen(file_name, intra_Al_N_data, intra_Al_N_flp_info, esp_charges_dir=False) for file_name in intra_Al_N_data[1]]
#to_pkl(intra_Al_N_descriptors, 'data/intra_Al_N_descriptors.pkl')
#intra_Al_N_descriptors = from_pkl('data/intra_Al_N_descriptors.pkl')
#intra_Al_N_dataset = dataset_builder(intra_Al_N_data, intra_Al_N_descriptors, 'data/intra_Al_N_reac_free_energies.csv', 'data/intra_Al_N_barr_free_energies.csv', 'data/intra_Al_N_feha_energies.csv', 'data/intra_Al_N_fepa_energies.csv', pkl_outfile='data/intra_Al_N_dataset', mode='test')
intra_Al_N_dataset = from_pkl('data/intra_Al_N_dataset.pkl')

val_data = [['inter_B_N', 'intra_B_N', 'intra_B_P', 'intra_Al_N'], [inter_B_N_dataset, intra_B_N_dataset, intra_B_P_dataset, intra_Al_N_dataset]]

###################
## Data Analysis ##
###################

# Predict reaction free energy using only FEHA

#k_fold_opt = k_fold_optimization(dataset.loc[:, ['FEHA']], dataset['free_ener_reac'], dataset['labels'], num_steps=20, num_repetitions=5, plot=True)

#reaction_free_energy_feha = pls_analysis(dataset['labels'], dataset.loc[:, ['FEHA']], dataset['free_ener_reac'], n_comp=1, cv_group_number=60, sig_figs=3, print_model=True)

# Prediction reaction free energy using all descriptors

#reaction_free_energy_all = pls_analysis(dataset['labels'], dataset.iloc[:, 5:], dataset['free_ener_reac'], n_comp=len(dataset.iloc[:, 5:].columns), cv_group_number=60, sig_figs=3, print_model=True)

# Predict reaction free energy using the three most important descriptors: d, FEHA, and FEPA

#reaction_free_energy_bestmodel = pls_analysis(dataset['labels'], dataset.loc[:, ['d', 'FEHA', 'FEPA']], dataset['free_ener_reac'], n_comp=3, cv_group_number=60, sig_figs=3, print_model=True)

## External validation

#reaction_energy_validation_results = model_validation(data=dataset, resp_var_name='free_ener_reac', val_data=val_data, desc=['d', 'FEHA', 'FEPA'], n_comp=reaction_free_energy_bestmodel['n_comp'], cv_group_number=reaction_free_energy_bestmodel['cv_group_number'])

# Predict reaction free energy using combinations of two descriptors

# reaction_free_energy_d_feha = pls_analysis(dataset['labels'], dataset.loc[:, ['d', 'FEHA']], dataset['free_ener_reac'], n_comp=2, cv_group_number=60, sig_figs=3, print_model=True)

# reaction_free_energy_feha_fepa = pls_analysis(dataset['labels'], dataset.loc[:, ['FEHA', 'FEPA']], dataset['free_ener_reac'], n_comp=2, cv_group_number=60, sig_figs=3, print_model=True)

# reaction_free_energy_d_fepa = pls_analysis(dataset['labels'], dataset.loc[:, ['d', 'FEPA']], dataset['free_ener_reac'], n_comp=2, cv_group_number=60, sig_figs=3, print_model=True)

#print('##################################')

# Predict barrier free energy using all descriptors

#k_fold_opt = k_fold_optimization(dataset.iloc[:, 5:], dataset['free_ener_barr'], dataset['labels'], num_steps=20, num_repetitions=5, plot=True)

#barr_free_energy_all = pls_analysis(dataset['labels'], dataset.iloc[:, 4:], dataset['free_ener_barr'], n_comp=len(dataset.iloc[:, 5:].columns), cv_group_number=60, sig_figs=3, print_model=True)

# Expand dataset with polynomial features

#dataset_expanded = data_expander('data/dataset.pkl', dataset.columns[5:], out_name='data/dataset_expanded')
dataset_expanded = from_pkl('data/dataset_expanded.pkl')

# Expnd dataset with polynomial features for testing set

#intra_B_N_dataset_expanded = data_expander('data/intra_B_N_dataset.pkl', intra_B_N_dataset.columns[5:], out_name='data/intra_B_N_dataset_expanded')
intra_B_N_dataset_expanded = from_pkl('data/intra_B_N_dataset_expanded.pkl')
#inter_B_N_dataset_expanded = data_expander('data/inter_B_N_dataset.pkl', inter_B_N_dataset.columns[5:], out_name='data/inter_B_N_dataset_expanded')
inter_B_N_dataset_expanded = from_pkl('data/inter_B_N_dataset_expanded.pkl')
#intra_B_P_dataset_expanded = data_expander('data/intra_B_P_dataset.pkl', intra_B_P_dataset.columns[5:], out_name='data/intra_B_P_dataset_expanded')
intra_B_P_dataset_expanded = from_pkl('data/intra_B_P_dataset_expanded.pkl')
#intra_Al_N_dataset_expanded = data_expander('data/intra_Al_N_dataset.pkl', intra_Al_N_dataset.columns[5:], out_name='data/intra_Al_N_dataset_expanded')
intra_Al_N_dataset_expanded = from_pkl('data/intra_Al_N_dataset_expanded.pkl')

val_data_expanded = [['inter_B_N', 'intra_B_N', 'intra_B_P', 'intra_Al_N'], [inter_B_N_dataset_expanded, intra_B_N_dataset_expanded, intra_B_P_dataset_expanded, intra_Al_N_dataset_expanded]]

# Predict barrier free energy using all descriptors and their polynomial features

#barr_free_energy_all_expanded = pls_analysis(dataset_expanded['labels'], dataset_expanded.iloc[:, 4:], dataset_expanded['free_ener_barr'], n_comp=len(dataset_expanded.iloc[:, 4:].columns), cv_group_number=60, sig_figs=3, print_model=True)

#Informative variable filter PLS iterative optimization

#info_var_barr = informative_variable_filter_pls(dataset_expanded.iloc[:, 4:], dataset_expanded['free_ener_barr'], n_random_models=10, max_predictors=15, out_name='data/informative_var_barr_results', comb_analysis=False)

#info_var_barr_meanthreshold = informative_variable_filter_pls(dataset_expanded.iloc[:, 4:], dataset_expanded['free_ener_barr'], n_random_models=10, max_predictors=15, out_name='data/informative_var_barr_results_meanthreshold', comb_analysis=False)

# info_var_barr_v2 = informative_variable_filter_pls_v2(
#     dataset_expanded.iloc[:, 4:],
#     dataset_expanded['free_ener_barr'],
#     n_random_sets=5,
#     n_random_models=10,
#     max_predictors=15,
#     out_name='data/informative_var_barr_results',
#     comb_analysis=False,
#     cv=60,
#     random_percentile=95
# )

# Optimized model for barrier free energy prediction using the selected informative variables

#barr_free_energy_opt = pls_analysis(dataset_expanded['labels'], dataset_expanded.loc[:, ['d$^2$', 'FEPA', 'FEHA', 'EF$_{la}$(ESP)', 'd', 'q$_{lb}$(NPA)$^2$', 'λ', 'Mw']], dataset_expanded['free_ener_barr'], n_comp=8, cv_group_number=60, sig_figs=3, print_model=True)

## External validation

#barr_free_energy_opt_validation_results = model_validation(data=dataset_expanded, resp_var_name='free_ener_barr', val_data=val_data_expanded, desc=['d$^2$', 'FEPA', 'FEHA', 'EF$_{la}$(ESP)', 'd', 'q$_{lb}$(NPA)$^2$', 'λ', 'Mw'], n_comp=barr_free_energy_opt['n_comp'], cv_group_number=barr_free_energy_opt['cv_group_number']).sort_values(by='Name')