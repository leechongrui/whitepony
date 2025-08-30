# Model dashboard for validation analysis
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# Load validation predictions

def dashboard_report(val_csv_path=None, output_dir=None):
	"""
	Generate dashboard metrics and mismatches for validation predictions.
	Args:
		val_csv_path (str): Path to validation_predictions.csv
		output_dir (str): Directory to save mismatches file
	Returns:
		dict: metrics summary
	"""
	if val_csv_path is None:
		val_csv_path = os.path.join(os.path.dirname(__file__), 'output/validation_predictions.csv')
	if output_dir is None:
		output_dir = os.path.join(os.path.dirname(__file__), 'output')
	os.makedirs(output_dir, exist_ok=True)

	df = pd.read_csv(val_csv_path)

	# 1. Output mismatches between true and predicted labels
	mismatches = df[df['true_label'] != df['predicted_label']]
	mismatches_path = os.path.join(output_dir, 'validation_mismatches.csv')
	mismatches.to_csv(mismatches_path, index=False)
	print(f"Mismatches saved to {mismatches_path} ({len(mismatches)} samples)")

	# 2. Print classification metrics
	y_true = df['true_label']
	y_pred = df['predicted_label']
	print("\n==== Metrics for HOTEL_MODEL Validation Set ====")
	print("\nClassification Report:")
	print(classification_report(y_true, y_pred, digits=4))
	print("Confusion Matrix:")
	print(confusion_matrix(y_true, y_pred, labels=['truthful', 'deceptive']))
	print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}\n")

	return {
		'classification_report': classification_report(y_true, y_pred, digits=4, output_dict=True),
		'confusion_matrix': confusion_matrix(y_true, y_pred, labels=['truthful', 'deceptive']).tolist(),
		'accuracy': accuracy_score(y_true, y_pred),
		'num_mismatches': len(mismatches),
		'mismatches_path': mismatches_path
	}

if __name__ == "__main__":
	dashboard_report()
