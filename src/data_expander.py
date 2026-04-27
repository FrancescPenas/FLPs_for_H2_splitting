import pandas as pd
import os
from utility_functions import from_pkl

def data_expander(df_original, expand_cols=None, out_name='df'):
    # This function takes a DataFrame and expands the specified columns by adding new features based on transformations of the original columns.
    path = os.getcwd()
    file_path = os.path.join(path, df_original)
    df = from_pkl(file_path)
    df_new = df.copy()
    
    if expand_cols is None:
        print("No columns specified for expansion. Returning original DataFrame.")
        return df_new
    
    expanded_cols = df.loc[:, expand_cols]
    for col in expand_cols:
        if col in df.columns:
            expanded_cols[col + '$^2$'] = df[col] ** 2
            # Uncomment below to enable more transformations
            # expanded_cols[col + '**-1'] = df[col] ** (-1)
            # expanded_cols[col + ' exp'] = np.exp(df[col])
            # expanded_cols[col + ' log'] = np.log(df[col])
    
    sorted_expanded = pd.DataFrame(expanded_cols).sort_index(axis=1)
    not_expanded_cols = df.columns.difference(expand_cols)
    df_new = pd.concat([df.loc[:, not_expanded_cols], sorted_expanded], axis=1)
    out_path = os.path.join(path, out_name)
    df_new.to_pickle(out_path + '.pkl')
    df_new.to_csv(out_path + '.csv', index=False)
    
    return df_new