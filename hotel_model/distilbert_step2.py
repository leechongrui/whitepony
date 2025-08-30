import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import torch
from transformers import DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments

# Load the CSV file directly from the path (replace with your actual file path)
df = pd.read_csv("../data/raw/fake reviews dataset.csv")  # Provide the correct file path

# Map labels to integers (Assuming 'computer_generated' and 'user_generated' as the labels)
label_map = {"CG": 0, "OR": 1}
df['generation_type'] = df['label'].map(label_map)

# Split into train + validation
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),
    df['generation_type'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['generation_type']
)



tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)



class ProductReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = ProductReviewDataset(train_encodings, train_labels)
val_dataset = ProductReviewDataset(val_encodings, val_labels)



model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)



training_args = TrainingArguments(
    output_dir='./results',
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

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

trainer.evaluate()

# Save the model
model.save_pretrained("/content/product_model")
tokenizer.save_pretrained("/content/product_model")
