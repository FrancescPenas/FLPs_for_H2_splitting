from src.dist_calc import dist_calc
import numpy as np

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
