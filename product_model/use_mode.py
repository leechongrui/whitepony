import os
import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Path to model and tokenizer
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'product_model_distilbert')

# Path to input file (change as needed)
INPUT_CSV = os.path.join(os.path.dirname(__file__), '../data/processed/model1_output.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'output', 'product_model_predictions.csv')

# Load model and tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Read input data
df = pd.read_csv(INPUT_CSV)
df = df.rename(columns={'text_': 'text'})
texts = df['text'].tolist()

# Tokenize
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')
encodings = {k: v.to(device) for k, v in encodings.items()}

# Predict
with torch.no_grad():
	outputs = model(**encodings)
	probs = torch.nn.functional.softmax(outputs.logits, dim=1)
	preds = torch.argmax(probs, dim=1).cpu().numpy()
	scores_CG = probs[:, 0].cpu().numpy()

# Map back to label names
label_map = {0: 'CG', 1: 'OR'}
df['predicted_label'] = [label_map[p] for p in preds]
df['score_CG'] = scores_CG

# Save results
df.to_csv(OUTPUT_CSV, index=False)
print(f"Predictions saved to {OUTPUT_CSV}")
