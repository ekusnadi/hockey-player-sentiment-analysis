from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

TRAINING_DATA_DIR = Path("data/training")
OUTPUT_DIR = "./roberta-sharks-finetuned"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Config 
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
TEST_SIZE = 0.15
SEED = 42

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Load data
input_file = TRAINING_DATA_DIR / "redacted_training_comments.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Could not find {input_file}")

df = pd.read_csv(input_file, encoding="latin-1")
df.columns = df.columns.str.strip()
df["Sentiment"] = df["Sentiment"].str.strip().str.lower()
df = df[df["Sentiment"].isin(LABEL2ID)].dropna(subset=["Comment"]).reset_index(drop=True)
df["label"] = df["Sentiment"].map(LABEL2ID)

train_df, val_df = train_test_split(df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED)
print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")
print("Label distribution (train):\n", train_df["Sentiment"].value_counts())

# Dataset
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

train_dataset = SentimentDataset(train_df["Comment"], train_df["label"])
val_dataset = SentimentDataset(val_df["Comment"], val_df["label"])

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# Train model
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LR,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

# Evaluation
preds = np.argmax(trainer.predict(val_dataset).predictions, axis=-1)
print("\nVal Accuracy:", accuracy_score(val_df["label"], preds))
print(classification_report(val_df["label"], preds, target_names=["negative", "neutral", "positive"]))

# Save model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nModel saved to {OUTPUT_DIR}")