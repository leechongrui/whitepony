# ingest.py
"""
Pull/download data, verify checksums, save to data/raw/
"""

import pandas as pd
import os

def load_hotel_spam_corpus(data_path=None):
	"""
	Load the hotel spam corpus CSV as a pandas DataFrame.
	Args:
		data_path (str): Path to the CSV file. If None, uses default location.
	Returns:
		pd.DataFrame: Loaded data.
	"""
	if data_path is None:
		data_path = os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			'data', 'raw', 'kaggle_hotel_spam_corpus.csv'
		)
	df = pd.read_csv(data_path)
	return df
