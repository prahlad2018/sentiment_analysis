# Custom ML Model as API (Flask Version)

# Step 1: Install Required Libraries
# pip install flask pandas numpy scikit-learn nltk vaderSentiment

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

# Step 2: Load and Prepare Model
original_df = pd.read_csv('C:/Users/ishit/OneDrive/Documents/Python/Office-Efficiency/user_feedback_dataset_corrected.csv')

# Preprocess Function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    clean_words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(clean_words)

original_df['clean_text'] = original_df['FEEDBACK_TEXT'].apply(preprocess)

# Label with VADER
analyzer = SentimentIntensityAnalyzer()
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

original_df['sentiment'] = original_df['clean_text'].apply(get_sentiment)

# Train Model
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(original_df['clean_text'])
y = original_df['sentiment']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Step 3: Create Flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or not isinstance(data, list):
        return jsonify({"error": "Invalid input format. Must be a list of records with 'id' and 'feedback_text'."}), 400

    results = []
    for record in data:
        record_id = record.get("id")
        feedback_text = record.get("feedback_text")

        if record_id is None or feedback_text is None:
            results.append({"id": record_id, "error": "Missing 'id' or 'feedback_text'."})
            continue

        clean_text = preprocess(feedback_text)
        vectorized = vectorizer.transform([clean_text])
        sentiment = clf.predict(vectorized)[0]

        results.append({
            "id": record_id,
            "feedback_text": feedback_text,
            "sentiment": sentiment
        })

    return jsonify(results)

# Step 4: Run Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False)
