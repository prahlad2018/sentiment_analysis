# Sentiment Analysis Using Hugging Face Pre-trained Model

# Step 1: Install Required Libraries
#The pip command is the Python package installer. It's used to install, upgrade, 
#and manage external Python libraries and packages from the Python Package Index (PyPI)
#pip install pandas transformers

# Step 2: Import Libraries
import pandas as pd
from transformers import pipeline

# Step 3: Load Data
# Replace with your CSV file path
df = pd.read_csv('user_feedback_dataset_corrected.csv')

# Step 4: Hugging Face Model (Pre-trained Sentiment Analysis)
sentiment_pipeline = pipeline("sentiment-analysis")

# Apply model on feedback texts
#1. Apply a function to each entry in the column FEEDBACK_TEXT.
#2. The applied function uses the sentiment_pipeline from Hugging Face to predict sentiment for each piece of text.
#3. sentiment_pipeline(x) returns a list of dictionaries with sentiment predictions, such as:[{'label': 'POSITIVE', 'score': 0.95}]
#[0]['label'] extracts the sentiment label ('POSITIVE' or 'NEGATIVE') from this list.
#The resulting sentiment label is assigned to a new column named huggingface_sentiment in the DataFrame.

df['huggingface_sentiment'] = df['FEEDBACK_TEXT'].apply(lambda x: sentiment_pipeline(x)[0]['label'])

# Evaluate Results
print("Hugging Face Sentiment Analysis Distribution:")
print(df['huggingface_sentiment'].value_counts())

# Save results
df.to_csv('huggingface_sentiment_analysis_results.csv', index=False)

print("Analysis complete. Results saved to 'huggingface_sentiment_analysis_results.csv'.")

