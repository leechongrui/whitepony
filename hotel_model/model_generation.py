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


# Split into train + validation and keep indices
train_indices, val_indices = train_test_split(
	range(len(df)),
	test_size=0.2,
	random_state=42,
	stratify=df['truth_value']
)
train_texts = df.loc[train_indices, 'text'].tolist()
val_texts = df.loc[val_indices, 'text'].tolist()
train_labels = df.loc[train_indices, 'truth_value'].tolist()
val_labels = df.loc[val_indices, 'truth_value'].tolist()

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
val_pred_scores = torch.nn.functional.softmax(torch.tensor(val_pred.predictions), dim=1).numpy()
val_pred_labels = np.argmax(val_pred_scores, axis=1)




# Append only the score_deceptive column to the original validation DataFrame using val_indices
original_val_df = df.iloc[val_indices].reset_index(drop=True)
original_val_df = original_val_df.copy()
original_val_df['score_deceptive'] = val_pred_scores[:, 1]

val_results_path = os.path.join(os.path.dirname(__file__), 'output/validation_predictions.csv')
original_val_df.to_csv(val_results_path, index=False)
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
