# train.py
"""
Trains models, saves artefacts (vectorisers, tokenizers, checkpoints) → models/
"""
# ...implementation to be added...
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ingest import load_hotel_spam_corpus
from preprocess import preprocess_dataframe

def train_model():
	# Load and preprocess data
	df = load_hotel_spam_corpus()
	df = preprocess_dataframe(df, text_column='text')

	# Label: 1 for deceptive (spam), 0 for truthful (ham)
	df['label'] = (df['deceptive'] == 'deceptive').astype(int)

	X = df['text']
	y = df['label']

	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

	# Pipeline: TF-IDF + Logistic Regression
	pipeline = Pipeline([
		('tfidf', TfidfVectorizer(max_features=5000)),
		('clf', LogisticRegression(max_iter=1000, random_state=42)),
	])

	pipeline.fit(X_train, y_train)

	# Save model
	model_path = os.path.join(os.path.dirname(__file__), 'spam_classifier.joblib')
	joblib.dump(pipeline, model_path)
	print(f"Model saved to {model_path}")

	return pipeline, X_val, y_val

if __name__ == "__main__":
	train_model()
