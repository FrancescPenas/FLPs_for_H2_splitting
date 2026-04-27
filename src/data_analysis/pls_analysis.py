import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd

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

def pls_analysis(names, X, y, n_comp, cv_group_number, sig_figs, print_model=False):
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
    normalized_coefficients = np.ravel(pls_normalized.coef_)
    intercept_normalized = np.mean(y_c_normalized) - np.dot(np.mean(X_normalized, axis=0), normalized_coefficients)

    # --- Normalized Model Expression (3 significant figures) ---
    terms_norm = [
    f"{format_sig_figs(coef, sig_figs)} * {name}" for coef, name in zip(normalized_coefficients, X.columns)
    ]
    model_expression_norm = " + ".join(terms_norm) + f" + {format_sig_figs(intercept_normalized, sig_figs)}"

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

    end_time = time.time()
    execution_time = end_time - start_time

    if print_model:
        print(f"\nModel expression (normalized coefficients):\n{model_expression_norm}")
        print(f"\nModel expression:\n{model_expression_non_norm}")

        # --- Print model stats ---
        print(f"\nr2: {score_c_normalized:.3f}")
        print(f"q2: {score_cv_normalized:.3f}")
        print(f"RMSE: {rmse_cv_normalized:.2f}")
        print(f"LV: {n_comp}")
        print('##################################')

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

    return {
        'cv_group_number': cv_group_number,
        'coef_normalized': normalized_coefficients,
        'coef_non_normalized': non_normalized_coefficients,
        'intercept_normalized': intercept_normalized,
        'intercept_non_normalized': intercept_non_normalized,
        'score_c_normalized': score_c_normalized,
        'score_cv_normalized': score_cv_normalized,
        'rmse_cv_normalized': rmse_cv_normalized,
        'n_comp': n_comp,
        'execution_time': execution_time,
        'results': predictions_table
    }