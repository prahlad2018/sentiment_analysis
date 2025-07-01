# Sentiment Analysis Using Custom ML Model

# Step 1: Install Required Libraries
# pip install pandas numpy scikit-learn nltk vaderSentiment
# pip install vaderSentiment

# Step 2: Import Libraries
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Step 3: Load Data
# Replace with your CSV file path
df = pd.read_csv('C:/Users/ishit/OneDrive/Documents/Python/Office-Efficiency/user_feedback_dataset_corrected.csv')

# Step 4: Text Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(clean_words)

df['clean_text'] = df['FEEDBACK_TEXT'].apply(preprocess)

# Step 5: Programmatic Labeling (VADER)
# The SentimentIntensityAnalyzer from the VADER (Valence Aware Dictionary and sEntiment Reasoner) module is designed to determine the sentiment of textual data.
# It assesses whether text expresses a positive, negative, or neutral tone, along with the strength of these sentiments.
#It works based on Lexicon-Based Approach:
#It uses a lexicon (dictionary of words) containing sentiment scores to calculate the sentiment polarity of text.
#Output Scores:
#Provides a detailed breakdown including:
#positive (pos): Probability of positive sentiment.
#negative (neg): Probability of negative sentiment.
#neutral (neu): Probability of neutral sentiment.
#compound: Combined score summarizing sentiment (-1 to +1).

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment'] = df['clean_text'].apply(get_sentiment)

# Step 6: Custom ML Model
#TfidfVectorizer converts textual data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency).
#Term Frequency (TF): Measures how often a word appears in a document.
#Inverse Document Frequency (IDF): Measures how common or rare a word is across all documents.
#TF-IDF assigns higher weights to important, rare words and lower weights to common words.

vectorizer = TfidfVectorizer(max_features=5000)
#This line transforms textual feedback into a numerical matrix of TF-IDF features that can be directly used to train a machine learning model.
#This applies the TF-IDF Vectorizer to the text data.
#It performs two actions:
#fit: Learns the vocabulary and the IDF (Inverse Document Frequency) from the text data.
#transform: Converts each document (text) into a TF-IDF-weighted numerical vector.
#X: The result is a sparse matrix of shape (n_samples, n_features):
#n_samples = number of text records.
#n_features = number of unique words used (up to max_features, like 5000).
#Each row in X is a vector representation of one piece of feedback.
#A sparse matrix is a matrix in which most of the elements are zero.

X = vectorizer.fit_transform(df['clean_text'])
y = df['sentiment']

#Training Data	Input-output pairs used to train the model
#Test Data	New data used to evaluate model performance. Purpose	Prevent overfitting and ensure generalization
#If you train and test on the same data, the model might memorize answers instead of learning patterns — leading to overfitting.
#Splitting the data ensures:
#Your model can generalize to new inputs.
#You get a realistic estimate of model performance.


#Splits data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
#Trains the model on training data
clf.fit(X_train, y_train)

#Makes predictions on test data
y_pred = clf.predict(X_test)

# Step 7: Evaluate Model
print("Custom ML Model Classification Report:")
# evaluate your machine learning model's performance by comparing the true labels (y_test) with the predicted labels (y_pred).
#High precision = when the model predicts a class, it is usually correct. Use precision when false positives are costly (e.g., marking a bad review as good).
# High recall = model successfully finds most of the actual items of that class. Use recall when false negatives are costly (e.g., missing a negative review in moderation).

print(classification_report(y_test, y_pred))

# Save results
df.to_csv('custom_ml_sentiment_analysis_results.csv', index=False)

print("Analysis complete. Results saved to 'custom_ml_sentiment_analysis_results.csv'.")
