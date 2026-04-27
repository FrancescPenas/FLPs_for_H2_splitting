import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
from utility_functions import from_pkl
import itertools
from tqdm import tqdm

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

def informative_variable_filter_pls(X, Y, n_random_models=10, max_predictors=15, out_name='informative_var_filter_results', comb_analysis = False):
    
    path = os.getcwd()
    name = os.path.basename(out_name)

    file_path = os.path.join(path, out_name + '.pkl')

    iteration = 1

    if not os.path.exists(file_path):
        converged = False
        while not converged:
            X_model = X.drop(columns=['labels'], errors='ignore')

            coef_original, coef_random = fit_pls_with_random_models(X_model, Y)
            mean_random_coef = np.mean(np.abs(coef_random), axis=0)
            importance_threshold = np.mean(mean_random_coef)
            mean_original_coef = np.mean(np.abs(coef_original), axis=0)

            all_coef = np.concatenate([mean_original_coef, mean_random_coef])
            all_coef_percent = (all_coef / max(all_coef) * 100).round(2)

            n_original = len(X_model.columns)
            n_random = len(all_coef_percent) - n_original
            random_names = [f"Random_{i+1}" for i in range(n_random)]
            all_names = np.concatenate([X_model.columns, random_names])

            df_coef = pd.DataFrame({
                'name': all_names,
                'value': all_coef_percent
            })

            dir_path = os.path.dirname(file_path)
            df_coef.to_csv(os.path.join(dir_path, f'iter_{iteration}_{name}.csv'), index=False)
            df_coef.to_pickle(os.path.join(dir_path, f'iter_{iteration}_{name}.pkl'))

            threshold_percent = (importance_threshold/max(all_coef)*100).round(2)
            important_predictors = mean_original_coef > importance_threshold

            iteration += 1

            if np.all(important_predictors) or sum(important_predictors) <= max_predictors:
                converged = True
            else:
                X = X_model.loc[:, important_predictors]

        # Ensure we do not exceed max_predictors
        importance_values_final = pd.Series(mean_original_coef, index=X_model.columns)
        if X_model.shape[1] > max_predictors:
            top_features = importance_values_final.nlargest(max_predictors).index
            X = X_model[top_features]

        importance_percent_final = (importance_values_final / max(importance_values_final) * 100).round(2).sort_values(ascending=False)

        optimal_components = optimize_components(X, Y)
        final_Y_pred = cross_val_predict(PLSRegression(n_components=optimal_components), X, Y, cv=60)
        final_rmse = np.sqrt(mean_squared_error(Y, final_Y_pred))
        final_q2 = 1 - (mean_squared_error(Y, final_Y_pred) / np.var(Y))

        print(f'Final RMSE: {final_rmse:.4f}')
        print(f'Final Q² Score: {final_q2:.4f}')

        if comb_analysis:
            indices = list(range(X.shape[1]))
            all_combinations = [
                list(combo)
                for r in range(1, min(X.shape[1], max_predictors) + 1)
                for combo in itertools.combinations(indices, r)
            ]
            results = []

            for combo in tqdm(all_combinations, desc='Processing Combinations', dynamic_ncols=True, ncols=80, colour="green"):
                selected_features = X.columns[list(combo)]

                optimal_components = min(optimize_components(X[selected_features], Y), len(selected_features))
                pls = PLSRegression(n_components=optimal_components)
                pls.fit(X[selected_features], Y)
                y_cv = cross_val_predict(pls, X[selected_features], Y, cv=60)

                r2 = r2_score(Y, pls.predict(X[selected_features]))
                q2 = r2_score(Y, y_cv)

                results.append([
                    list(selected_features),
                    len(selected_features),
                    optimal_components,
                    r2,
                    q2,
                    r2 + q2,
                    np.sqrt(mean_squared_error(Y, y_cv))
                ])

            results_df = pd.DataFrame(
                results,
                columns=['Combination', 'Num_Predictors', 'LVs', 'R2', 'Q2', 'R2+Q2', 'RMSE']
            )

            best_models = []
            for n in range(1, min(X.shape[1], max_predictors) + 1):
                subset = results_df[results_df['Num_Predictors'] == n]
                if not subset.empty:
                    best_models.append(subset.sort_values(by='R2+Q2', ascending=False).iloc[0])

            df = pd.DataFrame(best_models)

        else:
            df = 0

        if df is not 0:
            df.to_pickle(file_path)
            
            out_path = os.path.dirname(file_path)
            out_name = os.path.join(out_path, name + '.csv')
            df.to_csv(out_name, index=False)
            return df
    else:
        df = from_pkl(file_path)
        return df