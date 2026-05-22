from pathlib import Path
import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from textblob import TextBlob

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax


# Download VADER lexicon if not already downloaded
# nltk.download('vader_lexicon')

OUTPUT_DIR = Path("data/output")
TRAINING_DATA_DIR = Path("data/training")
input_file = TRAINING_DATA_DIR / "labeled_player_comments.csv"

if not input_file.exists():
    raise FileNotFoundError(f"Could not find {input_file}")

# Load the CSV dataset
df = pd.read_csv(input_file, encoding='latin-1')

# Initialize VADER sentiment analyzer
sia_base = SentimentIntensityAnalyzer()
sia = SentimentIntensityAnalyzer()

sia.lexicon.update({
    "filthy": 2.5,
    "fast": 1.0,
    "clutch": 2.5,
    "liability": -2.5,
    "turnover": -1.5,
    "soft": -1.5,
    "sniped": 2.0,
    "slow": -1.5,
    "lfg": 2.5,
    "brutal": -2.5,
    "terrible": -2.3,
    "awful": -2.7,
    "elite": 2.5,
    "lazy": -2.0,
    "washed": -2.5,
    "locked in": 2.3,
    "buzzing": 2.0,
    "dominant": 2.5,
    "unreal": 2.5,
    "garbage": -2.5,
    "useless": -2.5,
    "solid": 1.5,
    "invisible": -1.8
})

# Function to get sentiment label from VADER
def get_sentiment(text, model):
    scores = model.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def get_sentiment_textblob(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def save_mislabeled_comments(df, model):
    mislabeled = df[df['Sentiment'] != df[f'predicted_{model}']]
    print(f"\nNumber of mislabeled comments ({model}): {len(mislabeled)}/{len(df)}")
    if len(mislabeled) > 0:
        output_file = OUTPUT_DIR / f"mislabeled_comments_{model}.csv"
        mislabeled[['Comment', 'Sentiment', f'predicted_{model}']].to_csv(output_file, index=True)
        print(f"Saved {len(mislabeled)} mislabeled comments to {output_file.name}")
    else:
        print("No mislabeled comments.")

def save_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred, labels=['negative', 'neutral', 'positive'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'neutral', 'positive'])
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix ({model_name})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{model_name}.png")
    plt.close()
    print(f"Saved confusion matrix to confusion_matrix_{model_name}.png")


# Apply VADER to each text entry
df['predicted_vader_base'] = df['Comment'].apply(lambda x: get_sentiment(x, sia_base))
df['predicted_vader_aug'] = df['Comment'].apply(lambda x: get_sentiment(x, sia))

# Apply TextBlob to each text entry
df['predicted_textblob'] = df['Comment'].apply(get_sentiment_textblob)

# Calculate accuracy
accuracy_base = accuracy_score(df['Sentiment'], df['predicted_vader_base'])
accuracy_aug = accuracy_score(df['Sentiment'], df['predicted_vader_aug'])
accuracy_tb = accuracy_score(df['Sentiment'], df['predicted_textblob'])

# Generate classification report
report_base = classification_report(df['Sentiment'], df['predicted_vader_base'], target_names=['negative', 'neutral', 'positive'])
report_aug = classification_report(df['Sentiment'], df['predicted_vader_aug'], target_names=['negative', 'neutral', 'positive'])
report_textblob = classification_report(df['Sentiment'], df['predicted_textblob'], target_names=['negative', 'neutral', 'positive'])

# Print results
print(f"VADER Accuracy: {accuracy_base:.4f}")
print(f"VADER Accuracy (Augmented): {accuracy_aug:.4f}")
print(f"TextBlob Accuracy: {accuracy_tb:.4f}")
print("\nClassification Report (VADER):")
print(report_base)
print("\nClassification Report (VADER Augmented):")
print(report_aug)
print("\nClassification Report (TextBlob):")
print(report_textblob)

# Save results
save_mislabeled_comments(df, 'vader_base')
save_mislabeled_comments(df, 'vader_aug')
save_mislabeled_comments(df, 'textblob')

save_confusion_matrix(df['Sentiment'], df['predicted_vader_base'], 'vader_base')
save_confusion_matrix(df['Sentiment'], df['predicted_vader_aug'], 'vader_aug')
save_confusion_matrix(df['Sentiment'], df['predicted_textblob'], 'textblob')

# Initialize RoBERTa model and tokenizer
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
labels = {0: 'negative', 1: 'neutral', 2: 'positive'}

# Function to get sentiment label from RoBERTa
def get_sentiment_roberta(text):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return labels[np.argmax(scores)]

df['predicted_roberta'] = df['Comment'].apply(get_sentiment_roberta)
accuracy_roberta = accuracy_score(df['Sentiment'], df['predicted_roberta'])
report_roberta = classification_report(df['Sentiment'], df['predicted_roberta'], target_names=['negative', 'neutral', 'positive'])

# Print results for RoBERTa
print(f"\nRoBERTa Accuracy: {accuracy_roberta:.4f}")
print("\nClassification Report (RoBERTa):")
print(report_roberta)

save_mislabeled_comments(df, 'roberta')
save_confusion_matrix(df['Sentiment'], df['predicted_roberta'], 'roberta')
