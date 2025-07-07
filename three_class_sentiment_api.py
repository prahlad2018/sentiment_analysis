
# from transformers import pipeline

# sentiment = pipeline("sentiment-analysis", model="C:/Users/psing100/Development/office-efficiency/model_twitter/fine-tuned-twitter-roberta", tokenizer="C:/Users/psing100/Development/office-efficiency/model_twitter/fine-tuned-twitter-roberta")
# print(sentiment("The service was okay, not great but not bad."))

from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="C:/Users/psing100/Development/office-efficiency/model_twitter/fine-tuned-twitter-roberta",
    tokenizer="C:/Users/psing100/Development/office-efficiency/model_twitter/fine-tuned-twitter-roberta"
)

# Map model output labels to human-readable sentiment
label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive"
}

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    records = request.get_json()

    # Extract feedback texts
    texts = [record["feedback_text"] for record in records]

    # Run sentiment analysis
    results = sentiment_pipeline(texts)

    # Construct response
    response = []
    for record, result in zip(records, results):
        response.append({
            "id": record["id"],
            "feedback_text": record["feedback_text"],
            "predicted_sentiment": label_map.get(result["label"], "unknown"),
            "score": result["score"]
        })

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
