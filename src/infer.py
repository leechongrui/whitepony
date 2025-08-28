# infer.py
"""
Batch/CLI inference; reads text + metadata, outputs labels/scores/actions.
"""

import os
import pandas as pd
import joblib
from preprocess import preprocess_dataframe

def infer_google_reviews():
	# Paths
	data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'kaggle_google_restuarant_reviews.csv')
	model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier.joblib')
	output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'google_reviews_labelled.csv')

	# Load data
	df = pd.read_csv(data_path)
	df_proc = preprocess_dataframe(df, text_column='text')

	# Load model
	model = joblib.load(model_path)

	# Predict
	preds = model.predict(df_proc['text'])
	df['predicted_label'] = preds
	df['predicted_label'] = df['predicted_label'].map({0: 'ham', 1: 'spam'})


	# Save all predictions
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	df.to_csv(output_path, index=False)

	# Save only spam predictions
	spam_output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'google_reviews_spam_only.csv')
	df[df['predicted_label'] == 'spam'].to_csv(spam_output_path, index=False)
	print(f"Spam-only reviews saved to {spam_output_path}")

	# Count summary
	counts = df['predicted_label'].value_counts()
	total = len(df)
	print(f"Labelled reviews saved to {output_path}")
	print(f"\nPrediction counts:")
	for label in ['spam', 'ham']:
		print(f"{label}: {counts.get(label, 0)}")
	print(f"Total: {total}")

if __name__ == "__main__":
	infer_google_reviews()
