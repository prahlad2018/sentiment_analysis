import requests
import json

# Load input JSON
with open("C:/Users/ishit/OneDrive/Documents/Python/input_feedback_50.json", "r") as f:
    input_data = json.load(f)

# URL of your running Flask API
url = "http://127.0.0.1:5001/predict"

# Send POST request
response = requests.post(url, json=input_data)

# Print result
print("Status Code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))