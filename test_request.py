import requests
import json

# Load input JSON
with open("C:/Users/psing100/Development/sentiment_analysis/input_feedback_50.json", "r") as f:
    input_data = json.load(f)

# URL of your running Flask API
url = "http://10.99.73.146:5000/predict"

# Send POST request
response = requests.post(url, json=input_data)

# Print result
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))


