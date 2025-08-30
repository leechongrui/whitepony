# preprocess.py

import pandas as pd
import numpy as np

"""
Preprocess and clean sampled df
- 'resp' column - dict -> take text value out -> final value is string for resp column (DONE)

Feature engineering 
- 'text' column -> sentiment analysis score (DONE)
- 'text' length (DONE)
- 'response' - binary have or no response (DONE)
- 'pics' column - turn into binary: has_pics (DONE)

"""
from ingest import load_csv, save_dataframe
import ast
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# paths to sampled parquet and csv files ---- CHANGE ACCORDINGLY
sampled_parquet_path = r'C:\Users\Fabian\whitepony\data\interim\sampled_df.parquet'
sampled_csv_path = r'C:\Users\Fabian\whitepony\data\interim\sampled_df.csv'

# HELPER FUNCTION: function to create new column classifying 1 or 0 if empty or not: used on pics and resp columns 
def has_input(value: any) -> int:
    """
    accepts value from either pics or resp column to check, return 1 if theres input, 0 otherwise
    """
    # Handle NaN/None values
    if pd.isna(value) or value is None:
        return 0
    # Handle empty strings or just whitespace
    if isinstance(value, str) and value.strip() == '':
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

def create_binary_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    accepts df and column name to convert to binary,
    return new df with new binary column
    """
    # check if column name exists
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

# function to convert dict to string for resp column 
def extract_resp_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract text from 'resp' column and replace values in place.
    Skips rows where 'resp' is empty/null.
    accepts df and return new df with new resp column with extracted text
    """
    def get_text_from_resp(resp_value):
        """Extract text from response dict."""
        # skip rows where resp is empty/null
        if pd.isna(resp_value) or resp_value == '' or resp_value == '{}':
            return None
        
        try:
            # Parse string representation of dict
            if isinstance(resp_value, str):
                resp_dict = ast.literal_eval(resp_value)
            else:
                resp_dict = resp_value
            # Extract text field
            return resp_dict.get('text', None)
        # error handling
        except (ValueError, SyntaxError, AttributeError):
            return None
    
    # Create copy and extract text
    df['resp'] = df['resp'].apply(get_text_from_resp)
    return df

# function to generate text length column 
def add_word_count_column(df: pd.DataFrame, text_column: str, new_column_name: str) -> pd.DataFrame:
    """
    accepts df and text column name to add text length column, return new df with new text length column
    """
    df[new_column_name] = df[text_column].apply(
        lambda x: len(str(x).split()) if pd.notna(x) else 0
    )
    return df

# function to compute sentiment score for text column
nltk.download('vader_lexicon')
def add_sentiment_scores(df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
    """
    accepts df and text column name to add sentiment scores, return new df with new sentiment scores column
    """
    # initialise sentiment intensity analyzer
    sia = SentimentIntensityAnalyzer()
    # Apply VADER to each row
    df['sentiment_score'] = df[text_column].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    # classify into Positive / Neutral / Negative, purpose to detect extreme positive and negative reviews
    df['sentiment_label'] = df['sentiment_score'].apply(
        lambda x: 'positive' if x > 0.8 else ('negative' if x < -0.8 else 'neutral')
    )
    return df

if __name__ == "__main__":
    # load sampled df
    preprocess_df = load_csv(sampled_csv_path)

    # create binary column for pics and resp columns
    preprocess_df = create_binary_column(preprocess_df, 'pics')
    preprocess_df = create_binary_column(preprocess_df, 'resp')

    # extract text from resp column
    preprocess_df = extract_resp_text(preprocess_df)

    # add text length column
    preprocess_df = add_word_count_column(preprocess_df, 'text', 'word_count')

    # add sentiment scores column
    preprocess_df = add_sentiment_scores(preprocess_df)

    # save sampled df
    dest_path = r'C:\Users\Fabian\whitepony\data\interim\model_ready.csv'
    save_dataframe(preprocess_df, dest_path)
