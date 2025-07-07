

# pip install datasets

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd

# Load from local folder

# label_map = {
#     "LABEL_0": "Negative",
#     "LABEL_1": "Neutral",
#     "LABEL_2": "Positive"
# }

df = pd.read_csv("C:/Users/psing100/Development/office-efficiency/model_twitter/sentiment_data.csv")
dataset = Dataset.from_pandas(df)

model_path = "C:/Users/psing100/Development/office-efficiency/model_twitter"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

# Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)


# Create pipeline
# sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Training arguments
training_args = TrainingArguments(
    output_dir="C:/Users/psing100/Development/office-efficiency/model_twitter/results",
    # evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=500,
    save_total_limit=2,
    logging_dir="C:/Users/psing100/Development/office-efficiency/model_twitter/logs"
)



# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
)

trainer.train()


model.save_pretrained("C:/Users/psing100/Development/office-efficiency/model_twitter/fine-tuned-twitter-roberta")
tokenizer.save_pretrained("C:/Users/psing100/Development/office-efficiency/model_twitter/fine-tuned-twitter-roberta")
