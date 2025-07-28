import sys
import os

# Setup paths
sys.path.append(os.path.join(os.getcwd(), 'cdf francesc', 'FLPs', 'Python', 'code'))
#os.chdir(os.path.join(os.getcwd(), 'cdf francesc', 'FLPs', 'Python', 'code'))

from utility_functions_v2 import *
from sklearn.utils import shuffle
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import time
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from pls_analysis_v16 import pls_analysis

data = from_pkl('final_data_expanded.pkl')

# Thermo
#desc = ['d', 'FEHA', 'FEPA']
#resp_var_name = 'free_ener_reac'
#lv = 3

# Barr
desc = ['d$^2$', 'd', 'λ$^2$', 'λ', 'FEHA$^2$', 'FEPA', 'q_lb(NPA)$^2$', 'EP(NPA)$^2$', 'EP(NPA)']
resp_var_name = 'free_ener_barr'
lv = 9

# Barr no Attachment Energies
#desc = ['d$^2$', 'd', 'λ$^2$', 'λ', 'EF_lb(ESP)', 'EP(NPA)$^2$', 'EP(NPA)']
#resp_var_name = 'free_ener_barr'
#lv = 7

X = data.iloc[:, 4:].loc[:, desc]
Y = data[resp_var_name]
n_comp = lv
cv_group_number = 60

inter_B_N_data = from_pkl('inter_B_N_data.pkl')
intra_B_N_data = from_pkl('intra_B_N_data.pkl')
intra_B_P_data = from_pkl('intra_B_P_data.pkl')
intra_Al_N_data = from_pkl('intra_Al_N_data.pkl')
val_data = [['inter_B_N', 'intra_B_N', 'intra_B_P', 'intra_Al_N'], [inter_B_N_data, intra_B_N_data, intra_B_P_data, intra_Al_N_data]]

# shuffled y test
# Randomize y
#y_copy = Y.copy
y_randomized = shuffle(Y, random_state=42)

# PLS model for randomized data
pls_randomized = PLSRegression(n_components=n_comp, scale=False)
pls_randomized.fit(X, y_randomized)
y_c_randomized = pls_randomized.predict(X)
y_cv_randomized = cross_val_predict(pls_randomized, X, y_randomized, cv=cv_group_number)
score_c_randomized = r2_score(y_randomized, y_c_randomized)
score_cv_randomized = r2_score(y_randomized, y_cv_randomized)
rmse_cv_randomized = np.sqrt(mean_squared_error(y_randomized, y_cv_randomized))

print(f"\nShuffled y test:")
print(f"r2: {score_c_randomized:.4f}")
print(f"q2: {score_cv_randomized:.4f}")
print(f"RMSE: {rmse_cv_randomized:.4f}")
print(f"LV: {n_comp}")

# External set Validation
z = pls_analysis(data.iloc[:, 0], X, Y, n_comp, cv_group_number, 3)
coefs = z['coef_non_normalized']
inter = z['intercept_non_normalized']
columns = X.columns.tolist()

# Initialize a list to store results for all groups
validation_results = []

# Precompute coefficients and intercepts for efficiency
coefs_array = np.array(coefs)  # Convert coefficients to a numpy array for fast operations
inter_value = inter  # Intercept value

# Loop through validation groups
for i in range(len(val_data[0])):
    # Compute predictions using vectorized operations
    y_pred_list = (val_data[1][i][columns].values.dot(coefs_array) + inter_value).tolist()

    # Retrieve real values and names
    #y_real = val_data[1][i].iloc[:, 1].values  # reaction ener
    y_real = val_data[1][i].iloc[:, 2].values  # barriers
    names = val_data[1][i].iloc[:, 0].values  # Assumes first column contains names or IDs

    # Create a DataFrame for the group
    validation_df = pd.DataFrame({
        'Name': names,
        'Real Value': y_real,
        'Predicted Value': y_pred_list,
        'Error': abs(y_pred_list - y_real)
    })

    # Compute performance metrics
    score = r2_score(y_real, y_pred_list)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred_list))

    print(f"R2: {score:.4f} for {val_data[0][i]}")
    print(f"RMSE: {rmse:.4f} for {val_data[0][i]}")

    # Add a group column for clarity
    validation_df['Group'] = val_data[0][i]

    # Store the DataFrame in the results list
    validation_results.append(validation_df.round(1))

# Combine all group DataFrames into one for overview if needed
combined_results = pd.concat(validation_results, ignore_index=True)

# Display the combined results
pf(combined_results)