# preprocess.py

import pandas as pd
import numpy as np

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

def calculate_text_length(df, text_column='text', new_column_name='text_length'):
    """
    Calculate the length of text in a specified column and create a new column with the length.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    text_column : str, default='text'
        Name of the column containing text to measure
    new_column_name : str, default='text_length'
        Name of the new column to create with length values
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with new length column added
    """
    
    if text_column not in df.columns:
        print(f"Error: Column '{text_column}' not found in dataframe")
        return df
    
    # Create copy to avoid modifying original
    df_copy = df.copy()
    
    # Calculate text length, handling NaN/None values
    df_copy[new_column_name] = df_copy[text_column].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )
    
    # Print summary statistics
    total_rows = len(df_copy)
    avg_length = df_copy[new_column_name].mean()
    min_length = df_copy[new_column_name].min()
    max_length = df_copy[new_column_name].max()
    zero_length_count = (df_copy[new_column_name] == 0).sum()
    
    print(f"Created '{new_column_name}' column from '{text_column}':")
    print(f"  Average length: {avg_length:.1f} characters")
    print(f"  Min length: {min_length}")
    print(f"  Max length: {max_length}")
    print(f"  Rows with 0 length: {zero_length_count} ({zero_length_count/total_rows*100:.1f}%)")
    
    return df_copy

# Simple usage function specifically for 'text' column
def add_text_length_column(df):
    """
    Add a 'text_length' column to dataframe based on the 'text' column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with 'text' column
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with 'text_length' column added
    """
    
    return calculate_text_length(df, text_column='text', new_column_name='text_length')

# 

if __name__ == "__main__":
    # load sampled df
    sampled_df = load_csv(sampled_csv_path)
    

