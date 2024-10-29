# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------------------------------------------

def describe_and_suggest(df, cat_threshold=10, cont_threshold=10.0, count=False, transpose=False):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Input must be a pandas DataFrame, but received {type(df).__name__}.')
    
    print(f'{type(df)}')
    print(f"RangeIndex: {len(df)} entries, 0 to {len(df)-1}")
    print(f"Data columns (total {df.shape[1]} columns)")
    dtypes_summary = df.dtypes.value_counts()
    dtypes_string = ', '.join([f'{dtype.name}({count})' for dtype, count in dtypes_summary.items()])
    print(f"dtypes: {dtypes_string}")
    mem_usage = df.memory_usage(deep=True).sum() / 1024
    print(f"memory usage: {mem_usage:.1f} KB")
    print()
    total_missing_percentage = (df.isna().sum().sum())/len(df) *100
    print(f'Total Percentage of Null Values: {total_missing_percentage:.2f}%')
    
    if not isinstance(cat_threshold, int):
        raise TypeError(f'cat_threshold must be an int, but received {type(cat_threshold).__name__}.')
    if not isinstance(cont_threshold, float):
        raise TypeError(f'cont_threshold must be a float, but received {type(cont_threshold).__name__}.')

    num_rows = len(df)
    if num_rows == 0:
        raise ValueError('The DataFrame is empty.')

    data_type = df.dtypes
    null_count = df.notna().sum()
    missings = df.isna().sum()
    missings_perc = round(df.isna().sum() / num_rows * 100, 2)
    unique_values = df.nunique()
    cardinality = round(unique_values / num_rows * 100, 2)

    df_summary = pd.DataFrame({
        'Data Type': data_type,
        'Not-Null': null_count,
        'Missing': missings,
        'Missing (%)': missings_perc,
        'Unique': unique_values,
        'Cardinality (%)': cardinality
    })

    suggested_types = []

    for col in df.columns:
        card = unique_values[col]
        percent_card = card / num_rows * 100

        if card == 2:
            suggested_type = 'Binary'
        elif df[col].dtype == 'object':
            suggested_type ='Categorical'
        elif card < cat_threshold:
            suggested_type = 'Categorical'
        else:
            suggested_type = 'Numerical Continuous' if percent_card >= cont_threshold else 'Numerical Discrete'
        
        suggested_types.append(suggested_type)
    
    df_summary['Suggested Type'] = suggested_types

    if transpose:
        return df_summary.T
    return df_summary
