# Hugging Face Sentiment Analysis API (Flask Version)

# Step 1: Install dependencies
# pip install flask transformers torch certifi

from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os
import certifi

# ✅ Fix SSL certificate verification issue
# ✅ Load model locally after it's been downloaded once

model_name = "distilbert-base-uncased-finetuned-sst-2-english"


local_model_path = r"C:/Users/psing100/Development/sentiment_analysis/model"

model = AutoModelForSequenceClassification.from_pretrained(local_model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True)

sentiment_model = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)



# This creates a new Flask web application instance.
# __name__ tells Flask where to look for resources (like templates or static files).
app = Flask(__name__)

# This defines a route (URL endpoint) /predict that only accepts POST requests.
# When a POST request is made to /predict, the function predict_sentiment() will be called.
@app.route('/predict', methods=['POST'])
def predict_sentiment():
    # Extracts the JSON payload from the POST request.
    # Expects a list of records, each with an "id" and "feedback_text".
    data = request.get_json()

    # Checks if the input is missing or not a list.
    # If invalid, returns a 400 Bad Request with an error message.
    if not data or not isinstance(data, list):
        return jsonify({"error": "Request body must be a list of records with 'id' and 'feedback_text'."}), 400

    results = []
    for record in data:
        record_id = record.get("id")
        text = record.get("feedback_text")
        if record_id is None or text is None:
            results.append({"id": record_id, "error": "Missing 'id' or 'feedback_text'."})
            continue
        
        prediction = sentiment_model(text)[0]
        # sentiment = sentiment_model(text)[0]['label']
        # score = prediction['score']
        sentiment = prediction['label']
        score = prediction['score']

        results.append({
            "id": record_id,
            "feedback_text": text,
            "sentiment": sentiment,
             "score": score
        })

    return jsonify(results)

# Starts the Flask app on port 5000, accessible from any IP (0.0.0.0). 
# This ensures the Flask server only starts when you run the script directly, not when it's imported into another module (e.g., for testing).
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)