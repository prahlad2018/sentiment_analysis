# Hugging Face Sentiment Analysis API (Flask Version)

# Step 1: Install dependencies
# Run in terminal or notebook:
# !pip install flask transformers

# Step 2: Create the API using Flask
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
sentiment_model = pipeline("sentiment-analysis")

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()

    if not data or not isinstance(data, list):
        return jsonify({"error": "Request body must be a list of records with 'id' and 'feedback_text'."}), 400

    results = []
    for record in data:
        record_id = record.get("id")
        text = record.get("feedback_text")
        if record_id is None or text is None:
            results.append({"id": record_id, "error": "Missing 'id' or 'feedback_text'."})
            continue

        sentiment = sentiment_model(text)[0]['label']
        results.append({
            "id": record_id,
            "feedback_text": text,
            "sentiment": sentiment
        })

    return jsonify(results)

# Step 3: Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# Example input JSON:
# [
#     {"id": 1, "feedback_text": "I love this application!"},
#     {"id": 2, "feedback_text": "This is very frustrating to use."}
# ]

# Example curl command:
# curl -X POST http://localhost:5000/predict \
# -H "Content-Type: application/json" \
# -d '[{"id": 1, "feedback_text": "I love it!"}, {"id": 2, "feedback_text": "Not good."}]'
