import os
import itertools
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.cross_decomposition import PLSRegression

from utility_functions import from_pkl


def generate_random_block(X, seed=None):
    """
    Generate a random block by independently shuffling each column.
    This preserves the marginal distribution of each variable but
    destroys the original row-wise structure/correlations.
    """
    if seed is not None:
        np.random.seed(seed)

    R = X.to_numpy(copy=True)

    for j in range(R.shape[1]):
        R[:, j] = np.random.permutation(R[:, j])

    return R


def optimize_components(X, Y, cv=60):
    """
    Optimize the number of PLS components using CV RMSE.
    """
    max_components = min(X.shape[0], X.shape[1])

    rmses = []
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n, scale=False)
        y_cv = cross_val_predict(pls, X, Y, cv=cv)
        rmse = np.sqrt(mean_squared_error(Y, y_cv))
        rmses.append(rmse)

    return np.argmin(rmses) + 1


def fit_pls_with_random_models(X, Y, n_random_sets=5, n_random_models=10, cv=60):
    """
    Build PLS models using:
      - n_random_sets independently generated random blocks
      - n_random_models repetitions for each random block

    Returns arrays containing the normalized absolute coefficients
    for original and random variables from all fitted models.
    """
    X_values = X.to_numpy()
    n_samples, n_features = X.shape

    coef_original_all_models = []
    coef_random_all_models = []

    optimal_components = optimize_components(X, Y, cv=cv)
    optimal_components = min(optimal_components, n_features)

    for j in range(n_random_sets):
        # One new random set
        R_base = generate_random_block(X, seed=1000 + j)

        for i in range(n_random_models):
            # Extra reshuffling per model for stronger randomness
            np.random.seed(10000 + j * 100 + i)
            R = R_base.copy()
            for col in range(R.shape[1]):
                R[:, col] = np.random.permutation(R[:, col])

            X_aug = np.hstack((X_values, R))

            pls = PLSRegression(
                n_components=optimal_components,
                scale=False
            )
            pls.fit(X_aug, Y)

            impact_coef = np.abs(pls.coef_.flatten())
            impact_coef = impact_coef / np.sum(impact_coef)

            coef_original_all_models.append(impact_coef[:n_features])
            coef_random_all_models.append(impact_coef[n_features:])

    return np.array(coef_original_all_models), np.array(coef_random_all_models)


def informative_variable_filter_pls(
    X,
    Y,
    n_random_sets=5,
    n_random_models=10,
    max_predictors=15,
    out_name='informative_var_filter_results',
    comb_analysis=False,
    cv=60,
    random_percentile=95
):
    """
    Iterative informative variable filtering with PLS and random variables.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix. If a column named 'labels' exists, it is ignored.
    Y : array-like
        Response vector.
    n_random_sets : int
        Number of independently generated random variable sets.
    n_random_models : int
        Number of models built per random set.
    max_predictors : int
        Maximum number of predictors to keep.
    out_name : str
        Output basename (without extension).
    comb_analysis : bool
        If True, evaluate all combinations of selected variables.
    cv : int
        Number of CV folds.
    random_percentile : float
        Percentile used for the random-coefficient threshold.
    """
    path = os.getcwd()
    name = os.path.basename(out_name)
    file_path = os.path.join(path, out_name + '.pkl')

    iteration = 1

    if not os.path.exists(file_path):
        converged = False

        while not converged:
            X_model = X.drop(columns=['labels'], errors='ignore')

            coef_original, coef_random = fit_pls_with_random_models(
                X_model,
                Y,
                n_random_sets=n_random_sets,
                n_random_models=n_random_models,
                cv=cv
            )

            mean_original_coef = np.mean(np.abs(coef_original), axis=0)
            mean_random_coef = np.mean(np.abs(coef_random), axis=0)

            # Robust threshold from the whole distribution of random coefficients
            importance_threshold = np.percentile(
                np.abs(coef_random).ravel(),
                random_percentile
            )

            all_coef = np.concatenate([mean_original_coef, mean_random_coef])
            all_coef_percent = (all_coef / np.max(all_coef) * 100).round(2)

            n_original = len(X_model.columns)
            n_random = len(all_coef_percent) - n_original
            random_names = [f"Random_{i+1}" for i in range(n_random)]
            all_names = np.concatenate([X_model.columns, random_names])

            df_coef = pd.DataFrame({
                'name': all_names,
                'value': all_coef_percent
            })

            dir_path = os.path.dirname(file_path)
            df_coef.to_csv(
                os.path.join(dir_path, f'iter_{iteration}_{name}.csv'),
                index=False
            )
            df_coef.to_pickle(
                os.path.join(dir_path, f'iter_{iteration}_{name}.pkl')
            )

            threshold_percent = (
                importance_threshold / np.mean(all_coef) * 100
            ).round(2)

            important_predictors = mean_original_coef > importance_threshold

            print(f'Iteration {iteration}')
            print(f'Random threshold ({random_percentile}th percentile): {importance_threshold:.6f}')
            print(f'Threshold percent: {threshold_percent:.2f}%')
            print(f'Selected predictors: {np.sum(important_predictors)} / {len(important_predictors)}')

            iteration += 1

            if np.all(important_predictors) or np.sum(important_predictors) <= max_predictors:
                converged = True
            else:
                X = X_model.loc[:, important_predictors]

        # Final importance values from last iteration
        importance_values_final = pd.Series(
            mean_original_coef,
            index=X_model.columns
        )

        # Restrict to max_predictors if needed
        if X_model.shape[1] > max_predictors:
            top_features = importance_values_final.nlargest(max_predictors).index
            X = X_model[top_features]
        else:
            X = X_model.copy()

        importance_percent_final = (
            importance_values_final / np.max(importance_values_final) * 100
        ).round(2).sort_values(ascending=False)

        optimal_components = optimize_components(X, Y, cv=cv)
        optimal_components = min(optimal_components, X.shape[1])

        final_pls = PLSRegression(n_components=optimal_components, scale=False)
        final_Y_pred = cross_val_predict(final_pls, X, Y, cv=cv)

        final_rmse = np.sqrt(mean_squared_error(Y, final_Y_pred))
        final_q2 = 1 - (mean_squared_error(Y, final_Y_pred) / np.var(Y))

        print(f'Final RMSE: {final_rmse:.4f}')
        print(f'Final Q² Score: {final_q2:.4f}')
        print(f'Final selected predictors ({X.shape[1]}): {list(X.columns)}')

        if comb_analysis:
            indices = list(range(X.shape[1]))
            all_combinations = [
                list(combo)
                for r in range(1, min(X.shape[1], max_predictors) + 1)
                for combo in itertools.combinations(indices, r)
            ]

            results = []

            for combo in tqdm(
                all_combinations,
                desc='Processing Combinations',
                dynamic_ncols=True,
                ncols=80,
                colour='green'
            ):
                selected_features = X.columns[list(combo)]

                optimal_components = optimize_components(X[selected_features], Y, cv=cv)
                optimal_components = min(optimal_components, len(selected_features))

                pls = PLSRegression(n_components=optimal_components, scale=False)
                pls.fit(X[selected_features], Y)

                y_fit = pls.predict(X[selected_features])
                y_cv = cross_val_predict(pls, X[selected_features], Y, cv=cv)

                r2 = r2_score(Y, y_fit)
                q2 = r2_score(Y, y_cv)
                rmse = np.sqrt(mean_squared_error(Y, y_cv))

                results.append([
                    list(selected_features),
                    len(selected_features),
                    optimal_components,
                    r2,
                    q2,
                    r2 + q2,
                    rmse
                ])

            results_df = pd.DataFrame(
                results,
                columns=['Combination', 'Num_Predictors', 'LVs', 'R2', 'Q2', 'R2+Q2', 'RMSE']
            )

            best_models = []
            for n in range(1, min(X.shape[1], max_predictors) + 1):
                subset = results_df[results_df['Num_Predictors'] == n]
                if not subset.empty:
                    best_models.append(
                        subset.sort_values(by='R2+Q2', ascending=False).iloc[0]
                    )

            df = pd.DataFrame(best_models)

        else:
            df = importance_percent_final.reset_index()
            df.columns = ['Predictor', 'Importance_percent']

        df.to_pickle(file_path)

        out_path = os.path.dirname(file_path)
        csv_path = os.path.join(out_path, name + '.csv')
        df.to_csv(csv_path, index=False)

        return df

    else:
        df = from_pkl(file_path)
        return df