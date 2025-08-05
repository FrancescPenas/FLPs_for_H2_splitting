import sys
import os
sys.path.append(r''+os.getcwd()+'\FLPs\Python\code')
#os.chdir(r''+os.getcwd()+'\FLPs\Python\code')
from utility_functions import *
from gaussian_reader import gaussian_reader
from gaussian_freq_extractor import gaussian_freq_extractor
from dist_calc import dist_calc
import pandas as pd

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
    'ethy_dimethylaniline.out': 44,
    'me_aniline.log': 45,
    'cl_dimethylaniline.log': 46,
    'me_tetramethylbenzopiperidine.log': 47,
    'f_dimethylaniline.log': 48,
    'f_tetramethylbenzopiperidine.log': 49,
    'ipr_dimethylaniline.out': 50,
    'ipr_dimethylaniline.log': 50,
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

def h2_ts_dist_calc(file_name, coor_ts, freqs):
    freq_coor, freq_values = freqs
    freq_values = freq_values.astype(float).squeeze().tolist()

    num_negatives = sum([1 for num in freq_values if num < 0])
    if num_negatives == 0:
        print(f'No negative frequencies for {file_name}')
        return
    elif num_negatives >= 1:
        if num_negatives > 1:
            print(f'More than one negative frequency for {file_name}')
        # Identify significant modes
        im_freq = freq_coor[['AN', 'Freq 1']].copy()
        im_freq.columns = ['AN', 'X', 'Y', 'Z']
        im_freq['Module'] = np.sqrt(im_freq['X']**2 + im_freq['Y']**2 + im_freq['Z']**2)
        df_an_1 = im_freq[im_freq['AN'] == 1]
        top_two_modules = df_an_1.sort_values(by='Module', ascending=False).head(2)
        ts_h_labels = top_two_modules.index.tolist()
        h2_dist = dist_calc(coor_ts, ts_h_labels[0], ts_h_labels[1])
        return h2_dist

data_ts = from_pkl('input_ts_data.pkl')

indexes = []
h2_ts_dist_list = []
for file in data_ts[1]:
    if data_ts[2][file] in data_ts[0]:
        continue
    else:
        print(file)
        total_lines = gaussian_reader(file, 'input_ts')
        coor_ts = data_ts[2][file]
        freqs = gaussian_freq_extractor(total_lines, file)
        h2_dist = h2_ts_dist_calc(file, coor_ts, freqs)
        indexes += [file]
        h2_ts_dist_list += [h2_dist]
df = pd.DataFrame({
    'file_name': indexes,
    'h2_ts_dist': h2_ts_dist_list
})
df["label"] = df["file_name"].apply(lambda x: label_dict.get(x.replace('_ts1', ''), "Unknown"))
to_pkl(df, 'h_h_ts_dists.pkl')
df.to_csv('h_h_ts_dists.csv')
