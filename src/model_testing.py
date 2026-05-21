import numpy as np
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax


# Download VADER lexicon if not already downloaded
# nltk.download('vader_lexicon')

# Load the CSV dataset
df = pd.read_csv('./labeled_player_comments.csv', encoding='latin-1')

print("number of positive comments:", len(df[df['Sentiment'] == 'positive']))
print("number of neutral comments:", len(df[df['Sentiment'] == 'neutral']))
print("number of negative comments:", len(df[df['Sentiment'] == 'negative']))

# Initialize VADER sentiment analyzer
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
def get_sentiment(text):
    scores = sia.polarity_scores(str(text))
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

# Apply VADER to each text entry
df['predicted_vader'] = df['Comment'].apply(get_sentiment)

# Apply TextBlob to each text entry
df['predicted_textblob'] = df['Comment'].apply(get_sentiment_textblob)

# Calculate accuracy
accuracy = accuracy_score(df['Sentiment'], df['predicted_vader'])
accuracy_tb = accuracy_score(df['Sentiment'], df['predicted_textblob'])

# Generate classification report
report_vader = classification_report(df['Sentiment'], df['predicted_vader'], target_names=['negative', 'neutral', 'positive'])
report_textblob = classification_report(df['Sentiment'], df['predicted_textblob'], target_names=['negative', 'neutral', 'positive'])

# Print results
print(f"VADER Accuracy: {accuracy:.4f}")
print(f"TextBlob Accuracy: {accuracy_tb:.4f}")
print("\nClassification Report (VADER):")
print(report_vader)
print("\nClassification Report (TextBlob):")
print(report_textblob)

def show_mislabeled_comments(df, model):
    mislabeled = df[df['Sentiment'] != df[f'predicted_{model}']]
    print(f"\nNumber of mislabeled comments: {len(mislabeled)}/{len(df)}")
    if len(mislabeled) > 0:
        print("\nMislabeled Comments:")
        for idx, row in mislabeled.iterrows():
            print(f"Index {idx}: Text: '{row['Comment']}' | True: {row['Sentiment']} | Predicted: {row[f'predicted_{model}']}")
    else:
        print("No mislabeled comments.")

#show_mislabeled_comments(df, 'vader')

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

show_mislabeled_comments(df, 'roberta')