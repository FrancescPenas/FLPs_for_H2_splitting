import pandas as pd
from utility_functions import *
import sys
import os
sys.path.append(r'' + os.getcwd() + '\FLPs\Python\code')
os.chdir(r'' + os.getcwd() + '\FLPs\Python\code')

def data_expander(df, expand_cols=None, out_name='df'):
    """
    Expands selected columns by adding squared values and retains non-selected columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    expand_cols (list, optional): List of column names to expand. Expands all if None.

    Returns:
    pd.DataFrame: Expanded DataFrame with selected transformations.
    """
    df_new = df.copy()
    
    if expand_cols is None:
        expand_cols = df.columns  # Expand all columns if none are specified
    
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
    to_pkl(df_new, out_name + '.pkl')
    df_new.to_csv(out_name + '.csv', index=False)
    
    return df_new

df = from_pkl('final_data.pkl')

df_expanded = data_expander(df, df.columns[4:], out_name='final_data_expanded')
