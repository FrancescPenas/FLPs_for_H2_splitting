import sys
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Optional path setup (commented)
sys.path.append(os.path.join(os.getcwd(), 'cdf francesc', 'FLPs', 'Python', 'code'))
os.chdir(os.path.join(os.getcwd(), 'cdf francesc', 'FLPs', 'Python', 'code'))
from utility_functions_v2 import *

# Function to convert normalized coefficients to non-normalized
def convert_to_non_normalized_coefficients(normalized_coefficients, intercept_normalized, X_min, X_max):
    X_range = X_max - X_min
    non_normalized_coefficients = normalized_coefficients / X_range
    intercept_non_normalized = intercept_normalized - np.sum(normalized_coefficients * X_min / X_range)
    return non_normalized_coefficients, intercept_non_normalized

def format_sig_figs(value, sig_figs=3):
    if value == 0:
        return f"{0:.{sig_figs - 1}f}"
    # Determine the order of magnitude
    magnitude = int(np.floor(np.log10(abs(value))))
    decimals = sig_figs - magnitude - 1
    # Don't allow negative decimal places
    decimals = max(0, decimals)
    return f"{value:.{decimals}f}"

def pls_analysis(names, X, y, n_comp, cv_group_number, sig_figs):
    start_time = time.time()
    
    # Normalize X, not y
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)

    # Fit PLS model
    pls_normalized = PLSRegression(n_components=n_comp, scale=False)
    pls_normalized.fit(X_normalized, y)
    y_c_normalized = pls_normalized.predict(X_normalized)
    y_cv_normalized = cross_val_predict(pls_normalized, X_normalized, y, cv=cv_group_number)

    # Metrics
    score_c_normalized = r2_score(y, y_c_normalized)
    score_cv_normalized = r2_score(y, y_cv_normalized)
    rmse_cv_normalized = np.sqrt(mean_squared_error(y, y_cv_normalized))

    # Extract coefficients
    normalized_coefficients = pls_normalized.coef_[:, 0]
    intercept_normalized = np.mean(y_c_normalized) - np.dot(np.mean(X_normalized, axis=0), normalized_coefficients)

    # --- Normalized Model Expression (3 significant figures) ---
    terms_norm = [
    f"{format_sig_figs(coef, sig_figs)} * {name}" for coef, name in zip(normalized_coefficients, X.columns)
    ]
    model_expression_norm = " + ".join(terms_norm) + f" + {format_sig_figs(intercept_normalized, sig_figs)}"
    print(f"\nModel norm expression:\n{model_expression_norm}")

    # --- Convert to Non-Normalized Coefficients ---
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    non_normalized_coefficients, intercept_non_normalized = convert_to_non_normalized_coefficients(
        normalized_coefficients, intercept_normalized, X_min, X_max
    )

    # --- Non-Normalized Model Expression (3 significant digits) ---
    terms_non_norm = [
    f"{format_sig_figs(coef, sig_figs)} * {name}" for coef, name in zip(non_normalized_coefficients, X.columns)
    ]
    model_expression_non_norm = " + ".join(terms_non_norm) + f" + {format_sig_figs(intercept_non_normalized, sig_figs)}"

    print(f"\nModel expression:\n{model_expression_non_norm}")

    # --- Print model stats ---
    print(f"\nr2: {score_c_normalized:.3f}")
    print(f"q2: {score_cv_normalized:.3f}")
    print(f"RMSE: {rmse_cv_normalized:.2f}")
    print(f"LV: {n_comp}")

    # --- Predictions Table ---
    predictions_table = pd.DataFrame({
        'Name': names,
        'Actual Values': y,
        'Predicted Values (Score_c)': y_c_normalized.flatten(),
        'Difference (Score_c)': (y - y_c_normalized.flatten()),
        'Predicted Values (Score_cv)': y_cv_normalized.flatten(),
        'Difference (Score_cv)': (y - y_cv_normalized.flatten())
    }).round({
        'Actual Values': 1,
        'Predicted Values (Score_c)': 1,
        'Difference (Score_c)': 1,
        'Predicted Values (Score_cv)': 1,
        'Difference (Score_cv)': 1
    })

    end_time = time.time()
    print(f"\nExecution Time: {end_time - start_time:.2f} seconds")

    return {
        'coef_normalized': normalized_coefficients,
        'coef_non_normalized': non_normalized_coefficients,
        'intercept_normalized': intercept_normalized,
        'intercept_non_normalized': intercept_non_normalized,
        'score_c_normalized': score_c_normalized,
        'score_cv_normalized': score_cv_normalized,
        'rmse_cv_normalized': rmse_cv_normalized,
        'execution_time': end_time - start_time,
        'results': predictions_table
    }

# === Load data and model ===
df = from_pkl('final_data_expanded.pkl')

# Thermo
#desc = ['FEHA']
#resp_var = df['free_ener_reac']
#lv = 1

# Thermo
#desc = ['EF_la(ESP)', 'EF_la(NPA)',
#       'EF_lb(ESP)', 'EF_lb(NPA)',
#       'EP(ESP)', 'EP(NPA)', 'E_p(LA)',
#       'E_p(LB)', 'FEHA', 'FEPA',
#       'Mw', 'd', 'q_la(ESP)',
#       'q_la(NPA)', 'q_lb(ESP)', 'q_lb(NPA)',
#       'Φ', 'γ', 'λ']
#resp_var = df['free_ener_reac']
#lv = 19

# Thermo
#desc = ['d', 'FEHA', 'FEPA']
#resp_var = df['free_ener_reac']
#lv = 3

## Thermo
#desc = ['FEHA', 'FEPA']
#resp_var = df['free_ener_reac']
#lv = 2

## Thermo
#desc = ['d', 'FEHA']
#resp_var = df['free_ener_reac']
#lv = 2

# Thermo
#desc = ['d', 'FEPA']
#resp_var = df['free_ener_reac']
#lv = 2

## Thermo thresh opt
#desc = ["FEHA", "d", "FEPA", "q_la(ESP)", "EF_lb(ESP)", "EF_lb(NPA)", "EP(ESP)", "q_lb(NPA)"]
#resp_var = df['free_ener_reac']
#lv = 8

# Thermo exp thresh opt
desc = ['FEHA$^2$', 'FEHA', 'd$^2$', 'q_la(ESP)', 'EF_lb(ESP)$^2$', 'FEPA$^2$', 'EF_lb(NPA)$^2$']
resp_var = df['free_ener_reac']
lv = 6

## Barr
#desc = ['EF_la(ESP)', 'EF_la(NPA)',
#       'EF_lb(ESP)', 'EF_lb(NPA)',
#       'EP(ESP)', 'EP(NPA)', 'E_p(LA)',
#       'E_p(LB)', 'FEHA', 'FEPA',
#       'Mw', 'd', 'q_la(ESP)',
#       'q_la(NPA)', 'q_lb(ESP)', 'q_lb(NPA)',
#       'Φ', 'γ', 'λ']
#resp_var = df['free_ener_barr']
#lv = 19

## Barr exp
#desc = ['EF_la(ESP)', 'EF_la(ESP)$^2$', 'EF_la(NPA)', 'EF_la(NPA)$^2$',
#       'EF_lb(ESP)', 'EF_lb(ESP)$^2$', 'EF_lb(NPA)', 'EF_lb(NPA)$^2$',
#       'EP(ESP)', 'EP(ESP)$^2$', 'EP(NPA)', 'EP(NPA)$^2$', 'E_p(LA)',
#       'E_p(LA)$^2$', 'E_p(LB)', 'E_p(LB)$^2$', 'FEHA', 'FEHA$^2$', 'FEPA',
#       'FEPA$^2$', 'Mw', 'Mw$^2$', 'd', 'd$^2$', 'q_la(ESP)', 'q_la(ESP)$^2$',
#       'q_la(NPA)', 'q_la(NPA)$^2$', 'q_lb(ESP)', 'q_lb(ESP)$^2$', 'q_lb(NPA)',
#       'q_lb(NPA)$^2$', 'Φ', 'Φ$^2$', 'γ', 'γ$^2$', 'λ', 'λ$^2$']
#resp_var = df['free_ener_barr']
#lv = len(desc)

## Barr
#desc = ['d$^2$', 'd', 'λ$^2$', 'λ', 'FEHA$^2$', 'FEPA', 'q_lb(NPA)$^2$', 'EP(NPA)$^2$', 'EP(NPA)']
#resp_var = df['free_ener_barr']
#lv = 9

### Barr Threshold no Attachment Energies
##desc = ['d$^2$', 'd', 'λ$^2$', 'λ', 'EF_lb(ESP)', 'EP(NPA)$^2$', 'EP(NPA)']
##resp_var = df['free_ener_barr']
##lv = 7

data = df.iloc[:, 2:][desc]
pls_data = pls_analysis(df['labels'], data, resp_var, lv, 60, 3)

##Optimization lv
#list = []
#for lv in range(len(desc), 1, -1):
#    pls_data = pls_analysis(df['labels'], data, resp_var, lv, 60, 3)
#    list += [[lv, pls_data['score_c_normalized']+pls_data['score_cv_normalized']]]

#p(list)