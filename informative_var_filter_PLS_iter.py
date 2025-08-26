import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import os
from tqdm import tqdm
import itertools
import sys

sys.path.append(os.path.join(os.getcwd(), 'path', 'to', 'directory'))
os.chdir(os.path.join(os.getcwd(), 'path', 'to', 'directory'))

warnings.filterwarnings("ignore", category=FutureWarning, module='sklearn')
from utility_functions_v2 import *

# Max number of predictors allowed
max_predictors = 15  # Set the desired limit

#data = from_pkl('final_data.pkl')
#X = normalize_dataframe(data.iloc[:, 4:])
#Y = data.iloc[:, 2]
#n_random_models = 10
#out_file = 'opt_thermo_models_noexp.pkl'
#out_csv = 'opt_thermo_models_noexp.csv'

data = from_pkl('final_data_expanded.pkl')
X = normalize_dataframe(data.iloc[:, 5:])
Y = data.iloc[:, 1]
n_random_models = 10
out_file = 'opt_barr_models.pkl'
out_csv = 'opt_barr_models.csv'

def optimize_components(X, Y):
    max_components = min(X.shape[0], X.shape[1])
    rmses = [
        np.sqrt(mean_squared_error(Y, cross_val_predict(PLSRegression(n_components=n, scale=False), X, Y, cv=60)))
        for n in range(1, max_components + 1)
    ]
    return np.argmin(rmses) + 1

def fit_pls_with_random_models(X, Y, n_random_models=10):
    N, L = X.shape
    coef_original_all_models = []
    coef_random_all_models = []
    optimal_components = optimize_components(X, Y)
    
    for i in range(n_random_models):
        np.random.seed(42 + i)
        R = np.random.permutation(X.values)
        X_aug = np.hstack((X, R))
        pls = PLSRegression(n_components=optimal_components, scale=False).fit(X_aug, Y)
        impact_coef = np.abs(pls.coef_.flatten()) / np.sum(np.abs(pls.coef_.flatten()))
        coef_original_all_models.append(impact_coef[:L])
        coef_random_all_models.append(impact_coef[L:])
    
    return np.array(coef_original_all_models), np.array(coef_random_all_models)

iteration = 1

if not os.path.exists(out_file):
    converged = False
    while not converged:
        coef_original, coef_random = fit_pls_with_random_models(X, Y)
        mean_random_coef = np.mean(np.abs(coef_random), axis=0)
        importance_threshold = np.max(mean_random_coef)
        mean_original_coef = np.mean(np.abs(coef_original), axis=0)

        all_coef = np.concatenate([mean_original_coef, mean_random_coef])
        all_coef_percent = (all_coef/max(all_coef)*100).round(2)

        n_original = len(X.columns)
        n_random = len(all_coef_percent) - n_original
        random_names = [f"Random_{i+1}" for i in range(n_random)]
        all_names = np.concatenate([X.columns, random_names])

        df_coef = pd.DataFrame({
            'name': all_names,
            'value': all_coef_percent
        })

        to_pkl(df_coef, f'iter_{iteration}_barr_coefs.pkl')

        threshold_percent = (importance_threshold/max(all_coef)*100).round(2)
        important_predictors = mean_original_coef > importance_threshold

        iteration += 1

        if np.all(important_predictors) or sum(important_predictors) <= max_predictors:
            converged = True
        else:
            X = X.loc[:, important_predictors]

    # Ensure we do not exceed max_predictors
    importance_values_final = pd.Series(mean_original_coef, index=X.columns)
    if X.shape[1] > max_predictors:
        top_features = importance_values_final.nlargest(max_predictors).index
        X = X[top_features]

    importance_percent_final = (importance_values_final / max(importance_values_final) * 100).round(2).sort_values(ascending=False)

    optimal_components = optimize_components(X, Y)
    final_Y_pred = cross_val_predict(PLSRegression(n_components=optimal_components), X, Y, cv=60)
    final_rmse = np.sqrt(mean_squared_error(Y, final_Y_pred))
    final_q2 = 1 - (mean_squared_error(Y, final_Y_pred) / np.var(Y))

    print(f'Final RMSE: {final_rmse:.4f}')
    print(f'Final Q² Score: {final_q2:.4f}')

    comb_analysis = False #activate the combinatorial analysis section

    if comb_analysis == True:
        indices = list(range(X.shape[1]))
        all_combinations = [list(combo) for r in range(1, min(X.shape[1], max_predictors) + 1) for combo in itertools.combinations(indices, r)]
        results = []
    
        for combo in tqdm(all_combinations, desc='Processing Combinations', dynamic_ncols=True, ncols=80, colour="green"):
            selected_features = X.columns[list(combo)]
        
            if len(selected_features) == 0:
                continue
        
            optimal_components = min(optimize_components(X[selected_features], Y), len(selected_features))
            pls = PLSRegression(n_components=optimal_components)
            pls.fit(X[selected_features], Y)
            y_cv = cross_val_predict(pls, X[selected_features], Y, cv=60)
        
            results.append([
                list(selected_features), len(selected_features), optimal_components,
                r2_score(Y, pls.predict(X[selected_features])),
                r2_score(Y, y_cv), r2_score(Y, pls.predict(X[selected_features])) + r2_score(Y, y_cv),
                np.sqrt(mean_squared_error(Y, y_cv))
            ])
    
        results_df = pd.DataFrame(results, columns=['Combination', 'Num_Predictors', 'LVs', 'R2', 'Q2', 'R2+Q2', 'RMSE'])
        best_models = []
        for n in range(1, min(X.shape[1], max_predictors) + 1):
            subset = results_df[results_df['Num_Predictors'] == n]
            if not subset.empty:
                best_models.append(subset.sort_values(by='R2+Q2', ascending=False).iloc[0])

        df = pd.DataFrame(best_models)
        df.to_pickle(out_file)
    else:
        df = from_pkl(out_file)
    
    df.to_csv(out_csv, index=False)
