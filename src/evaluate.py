# evaluate.py
"""
Metrics, PR curves, error buckets → files under reports/
"""

import os
import joblib
from sklearn.metrics import classification_report, accuracy_score
from train import train_model

def evaluate_model():
	# Train model and get validation data
	model, X_val, y_val = train_model()
	y_pred = model.predict(X_val)
	print("Validation Accuracy:", accuracy_score(y_val, y_pred))
	print("\nClassification Report:\n", classification_report(y_val, y_pred, target_names=["ham", "spam"]))

if __name__ == "__main__":
	evaluate_model()
