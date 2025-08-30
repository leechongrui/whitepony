# ingest.py
"""
- Download and store carlifornia reviews and business metadata datasets as dataframe
- drop irrelevant metadata columns: address, latitude, longitude, URL (address data covered by gmap_id)
- filter review dataset rows where text is empty or non-english
- Extract unique categories from the category column in the business metadata
- reclassify categories in metadata category column into broader categories
- merge review dataset and business metadata on gmap_id (one-to-many merge - a business can have many reviews)
- Stratified sampling: for each unique business cat, sample 500 - 1000 reviews. if cat < 500 reviews, take all reviews from that cat.
- generate final dataset as csv and parquet files
"""
import gzip
import json
import pandas as pd
from typing import Union, List, Optional, Set
import os
import re

# paths to review and business metadata datasets. -------- CHANGE ACCORDINGLY
review_filepath = r"C:\Users\Fabian\whitepony\data\raw\review-California_10.json.gz"
metadata_filepath = r"C:\Users\Fabian\whitepony\data\raw\meta-California.json.gz"

# function to load csv files into dataframe
def load_csv(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        print(f"CSV file successfully loaded from {file_path}")
        print(df.info())
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs

# function to load parquet files into dataframe
def load_parquet(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(file_path)
        print(f"Parquet file successfully loaded from {file_path}")
        print(df.info())
        return df
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return pd.DataFrame()  # Return empty DataFrame if error occurs

# function accepting file path to .json.gz and return as a dataframe. consider efficiency since dataset is large. 
def load_json_gz(file_path: str, progress_interval: int = 1000000, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    accepts file path to .json.gz and returns a pd.DataFrame
    """
    data = []
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            # check json line not empty before loading
            if line.strip():
                data.append(json.loads(line))
            # progress tracker  
            if i % progress_interval == 0:
                print(f"Loaded {i:,} rows")
            # check if max_rows is set and reached
            if max_rows and i >= max_rows:
                break
    print(f"Total rows loaded: {len(data):,} rows")
    return pd.DataFrame(data)

# function accepting a pd.DataFrame and column names to drop. return new df with columns dropped
def drop_columns(df: pd.DataFrame, columns: Union[str, List[str]]) -> pd.DataFrame:
    """
    accepts a pd.DataFrame and column name OR list of column names to drop
    returns a new df with specified columns dropped
    """
    # convert to list if single string
    if isinstance(columns, str):
        columns = [columns]
    # check which columns exist
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]    
    # Print missing columns
    if missing_cols:
        print(f"Columns not found (skipped): {missing_cols}")   
    # Drop only existing columns
    if existing_cols:
        return df.drop(columns=existing_cols)
    else:
        return df

# function to check if text is english 
def is_english(text: str) -> bool:
    """
    Check if the given text is likely to be in English. uses simple regex to check if the text contains non-ASCII characters.
    """
    # Return True if text contains only ASCII characters (i.e., likely English)
    return bool(re.match(r'^[\x00-\x7F]*$', text))  # ASCII range is from 0 to 127

# function to filter rows where text is empty or non-english
def filter_non_english_or_empty(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    accepts df and column; filters rows where the specified column is empty or contains non-English text.
    returns new df with filtered rows
    """
    filtered_df = df[df[column].notna() 
                    & df[column].apply(lambda x: is_english(str(x))) 
                    & (df[column] != "")]
    return filtered_df

# function accepting a pd.DataFrame, then provide basic details about the dataframe
def inspect_df(df: pd.DataFrame) -> None:
    """
    accepts a pd.DataFrame and provides basic details about the dataframe
    ADJUST ACCORDINGLY
    """
    print("First row values:")
    print(df.iloc[0]) 
    print("\nColumn data types:")
    print(df.dtypes) 
    print("\nDataframe info (general overview):")
    print(df.info()) 

# function to extract unique categories from category column in business metadata 
def extract_unique_categories(df: pd.DataFrame, column: str = 'category') -> Set[str]:
    """
    Extract unique values from a column containing list objects.
    Case-insensitive.
    returns set of unique string categories
    """
    unique_values = set()
    for item_list in df[column]:
        if isinstance(item_list, list) and item_list:
            for item in item_list:
                if pd.notna(item):
                    unique_values.add(item.lower())
    return unique_values

# function to recategorise business categories into broader categories
def reclassify_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    reclassify business metadata category column into broader categories
    accepts DataFrame with 'category' column containing lists of narrow categories
    returns DataFrame with modified 'category' column containing broad categories
    """

    # Get all unique narrow categories first
    unique_categories = extract_unique_categories(df, 'category')

    # Create direct mapping from narrow to broad categories
    narrow_to_broad = {}
    
    # Keyword patterns for mapping
    keywords = {
        'food_dining': ['restaurant', 'cafe', 'bar', 'bakery', 'pizza', 'grill', 'buffet', 'takeaway', 'brewery', 'pub', 'diner', 'bistro'],
        'retail_shopping': ['store', 'shop', 'market', 'outlet', 'mall', 'supermarket', 'wholesaler', 'supplier', 'dealer'],
        'automotive': ['auto', 'car', 'vehicle', 'motorcycle', 'garage', 'tire', 'gas station', 'fuel', 'repair', 'dealer'],
        'healthcare': ['doctor', 'hospital', 'clinic', 'medical', 'dentist', 'pharmacy', 'health', 'surgeon', 'therapy'],
        'education': ['school', 'university', 'college', 'academy', 'institute', 'training', 'education', 'library'],
        'recreation': ['club', 'gym', 'park', 'theater', 'museum', 'sports', 'golf', 'cinema', 'entertainment'],
        'professional_services': ['lawyer', 'attorney', 'consultant', 'accountant', 'agency', 'office', 'engineer', 'service'],
        'government': ['government', 'federal', 'municipal', 'public', 'military', 'police', 'court', 'department'],
        'construction': ['contractor', 'construction', 'builder', 'roofing', 'plumbing', 'electrical', 'repair'],
        'manufacturing': ['manufacturer', 'factory', 'plant', 'industry', 'production', 'machinery'],
        'accommodation': ['hotel', 'lodge', 'accommodation', 'bed & breakfast', 'camping', 'resort'],
        'religious': ['church', 'mosque', 'synagogue', 'temple', 'religious']
    }
    
    # Build mapping dictionary once 
    for category in unique_categories:
        mapped = False
        for broad_cat, keyword_list in keywords.items():
            # check if any word in narrow category is a keyword in keyword_list
            if any(keyword in category for keyword in keyword_list):
                narrow_to_broad[category] = broad_cat
                mapped = True
                break
        if not mapped:
            narrow_to_broad[category] = 'other'
    
    # Fast lookup function; accepts a list of narrow categories (business metadata category value)
    def map_category(cat_list):
        # if row has no categories or not a list, classify as 'other'
        if not cat_list or not isinstance(cat_list, list):
            return 'other'
        # Return first matching category (convert to lowercase for lookup)
        for cat in cat_list:
            if pd.notna(cat):
                return narrow_to_broad.get(cat.lower(), 'other')
        return 'other'
    
    # Apply mapping
    df['category'] = df['category'].apply(map_category)
    return df

# function to stratify merged dataframe by business category
def stratified_sample_by_category(df: pd.DataFrame, sample_size: int = 500) -> pd.DataFrame:
    """
    accepts df then stratified sample of specified sample size by category column
    returns df with sampled rows from each category
    """
    sampled_dfs = []
    # Group by category and sample from each group
    for category in df['category'].unique():
        category_data = df[df['category'] == category]     
        # Sample up to sample_size rows or all if fewer available
        n_sample = min(sample_size, len(category_data))
        sampled = category_data.sample(n=n_sample, random_state=42)
        sampled_dfs.append(sampled)
    # Combine all samples
    result = pd.concat(sampled_dfs, ignore_index=True)
    print(f"Sampled {len(result):,} total rows from {len(sampled_dfs)} categories")
    return result

def save_dataframe(df: pd.DataFrame, dest_path: str) -> None:
    """
    accepts pd.DataFrame and dest_path then saves a DataFrame to a file at the specified destination path.
    Supports CSV (.csv) and Parquet (.parquet) formats.
    Creates parent directories if they do not exist.
    """
    # Ensure parent directories exist
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # Determine file format based on extension
    ext = os.path.splitext(dest_path)[1].lower()
    # save df
    if ext == '.csv':
        df.to_csv(dest_path, index=False)
    elif ext == '.parquet':
        df.to_parquet(dest_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use '.csv' or '.parquet'.")
    print(f"DataFrame successfully saved to {dest_path}")

if __name__ == "__main__":
    # load datasets ----- ADJUST MAX ROWS ACCORDINGLY
    metadata_df = load_json_gz(metadata_filepath)
    review_df = load_json_gz(review_filepath, max_rows=10000000)

    # drop irrelevant columns from metadata
    metadata_df = drop_columns(metadata_df, 
    ['name', 'address', 'latitude', 'longitude', 'url', 'relative_results', 'state', 'description', 'MISC', 'price'])
    review_df = drop_columns(review_df, ['name', 'time'])

    # filter rows where 'text' column is empty or non-english
    review_df = filter_non_english_or_empty(review_df, 'text')

    # reclassify categories in metadata
    metadata_df = reclassify_categories(metadata_df)

    # merge review and metadata
    merged_df = review_df.merge(metadata_df, on='gmap_id', how='inner', suffixes=('_review', '_metadata'))

    # stratified sampling ----- ADJUST SAMPLE SIZE PER CAT
    sampled_df = stratified_sample_by_category(merged_df, sample_size=5000)

    # save sampled df
    save_dataframe(sampled_df, r'C:\Users\Fabian\whitepony\data\interim\sampled_df.csv')
    save_dataframe(sampled_df, r'C:\Users\Fabian\whitepony\data\interim\sampled_df.parquet')
