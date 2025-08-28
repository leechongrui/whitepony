# preprocess.py
"""
Cleaning, normalisation, splits, feature building → data/processed/
"""
# ...implementation to be added...
import re
import string

import pandas as pd

def clean_text(text):
	"""
	Basic text cleaning: lowercase, remove punctuation, digits, and extra spaces.
	"""
	text = text.lower()
	text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
	text = re.sub(r'\d+', '', text)
	text = re.sub(r'\s+', ' ', text).strip()
	return text

def preprocess_dataframe(df, text_column='text'):
	"""
	Apply text cleaning to a DataFrame column.
	"""
	df = df.copy()
	df[text_column] = df[text_column].astype(str).apply(clean_text)
	return df
