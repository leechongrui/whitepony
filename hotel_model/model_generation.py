import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from model_dashboard import dashboard_report
'''
Hotel Model - Model Generation
'''

# Load CSV (use hotel spam corpus from data/raw)
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'kaggle_hotel_spam_corpus.csv')
df = pd.read_csv(csv_path)

# Map labels to integers
label_map = {"truthful": 0, "deceptive": 1}
df['truth_value'] = df['deceptive'].map(label_map)

# Split into train + validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
	df['text'].tolist(),
	df['truth_value'].tolist(),
	test_size=0.2,
	random_state=42,
	stratify=df['truth_value']
)

# Tokenization of text
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

# Create a custom dataset
class HotelDataset(torch.utils.data.Dataset):
	def __init__(self, encodings, labels):
		self.encodings = encodings
		self.labels = labels
	def __len__(self):
		return len(self.labels)
	def __getitem__(self, idx):
		item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
		item['labels'] = torch.tensor(self.labels[idx])
		return item

train_dataset = HotelDataset(train_encodings, train_labels)
val_dataset = HotelDataset(val_encodings, val_labels)

# Initialize model
model = DistilBertForSequenceClassification.from_pretrained(
	"distilbert-base-uncased",
	num_labels=2
)

# Define training arguments
training_args = TrainingArguments(
	output_dir='./output/checkpoints',
	num_train_epochs=3,
	per_device_train_batch_size=16,
	per_device_eval_batch_size=32,
	warmup_steps=50,
	weight_decay=0.01,
	logging_dir='./logs',
	logging_steps=20,
	eval_strategy="epoch",
	save_strategy="epoch"
)

# Initialize Trainer
trainer = Trainer(
	model=model,
	args=training_args,
	train_dataset=train_dataset,
	eval_dataset=val_dataset
)

# Train and evaluate the model
trainer.train()
eval_results = trainer.evaluate()


''' 
TO VERIFY THE OUTPUT
'''

# Get predictions on validation set
val_pred = trainer.predict(val_dataset)
val_pred_labels = np.argmax(val_pred.predictions, axis=1)

# Save validation results to CSV
val_results_df = pd.DataFrame({
	'text': val_texts,
	'true_label': val_labels,
	'predicted_label': val_pred_labels
})
val_results_df['true_label'] = val_results_df['true_label'].map({0: 'truthful', 1: 'deceptive'})
val_results_df['predicted_label'] = val_results_df['predicted_label'].map({0: 'truthful', 1: 'deceptive'})
val_results_path = os.path.join(os.path.dirname(__file__), 'output/validation_predictions.csv')
val_results_df.to_csv(val_results_path, index=False)
print("Evaluation results:", eval_results)
print(f"Validation predictions saved to {val_results_path}")
dashboard_report(val_csv_path=val_results_path)

''' 
Saving the model locally
'''
save_dir = os.path.join(os.path.dirname(__file__), 'hotel_model_distilbert')
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"Model and tokenizer saved to {save_dir}")
