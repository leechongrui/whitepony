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
sampled_csv_path = r'C:\Users\Fabian\whitepony\data\interim\sampled_df.csv'

# function to create new column classifying 1 or 0 if empty or not: used on pics and resp columns 

# function to convert dict to string for resp column 

# function to generate text length column 

# 

if __name__ == "__main__":
    # load sampled df
    sampled_df = load_csv(sampled_csv_path)
    

