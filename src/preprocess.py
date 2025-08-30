# preprocess.py
"""
Preprocess and clean sampled df
- 'pics' column - turn into binary: has_pics
- 'resp' column - dict -> take text value out -> final value is string for resp column 

Feature engineering 
- 'text' column -> sentiment analysis score (NOT)
- 'text' length
- 'user_id' and 'rating' - spot reviews from users with high number of extreme ratings (NOT)
- 'text' - extract promotional or irrelevant keywords like "sale" etc (NOT)
- 'response' - binary have or no response 

"""
from ingest import load_csv, load_parquet

# paths to sampled parquet and csv files
sampled_parquet_path = r'C:\Users\Fabian\whitepony\data\interim\sampled_df.parquet'
sampled_csv_path = r'C:\Users\Wang Song\Downloads\Tiktok hackathon\whitepony\data\interim'

# function to create new column classifying 1 or 0 if empty or not: used on pics and resp columns 

import pandas as pd
import numpy as np

def has_input(value):
    """
    Check if a cell has meaningful input (1) or is empty/null (0).
    
    Parameters:
    -----------
    value : any
        The value to check
    
    Returns:
    --------
    int : 1 if has input, 0 if no input
    """
    
    # Handle NaN/None values
    if pd.isna(value) or value is None:
        return 0
    
    # Handle empty strings or just whitespace
    if isinstance(value, str) and value.strip() == '':
        return 0
    
    # Handle numeric zeros
    if isinstance(value, (int, float)) and value == 0:
        return 0
    
    # Handle string representations of zero or empty structures
    if isinstance(value, str):
        cleaned_value = value.strip()
        if cleaned_value in ['0', '[]', '{}', '()', 'null', 'NULL', 'None']:
            return 0
    
    # Handle empty lists
    if isinstance(value, list) and len(value) == 0:
        return 0
    
    # If we get here, the value has meaningful content
    return 1

def create_binary_column(df, column_name):
    """
    Create a binary indicator column for any given column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column_name : str
        Name of the column to convert to binary
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with new binary column added
    """
    
    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in dataframe")
        return df
    
    # Create copy to avoid modifying original
    df_copy = df.copy()
    
    # Create binary column
    binary_col_name = f"{column_name}_binary"
    df_copy[binary_col_name] = df_copy[column_name].apply(has_input)
    
    # Print summary
    total = len(df_copy)
    has_input_count = df_copy[binary_col_name].sum()
    no_input_count = total - has_input_count
    
    print(f"Created '{binary_col_name}' from '{column_name}':")
    print(f"  Has input (1): {has_input_count} ({has_input_count/total*100:.1f}%)")
    print(f"  No input (0): {no_input_count} ({no_input_count/total*100:.1f}%)")
    
    return df_copy

def create_pics_resp_binary(df):
    """
    Create binary columns for pics and resp columns specifically.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with pics_binary and resp_binary columns
    """
    
    df_result = df.copy()
    
    # Create pics_binary column
    if 'pics' in df.columns:
        df_result = create_binary_column(df_result, 'pics')
    else:
        print("Warning: 'pics' column not found")
    
    # Create resp_binary column  
    if 'resp' in df.columns:
        df_result = create_binary_column(df_result, 'resp')
    else:
        print("Warning: 'resp' column not found")
    
    return df_result



# function to convert dict to string for resp column 

# function to generate text length column 

# 

if __name__ == "__main__":
    # load sampled df
    sampled_df = load_csv(sampled_csv_path)
    

