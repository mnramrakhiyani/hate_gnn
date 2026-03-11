from transformers import pipeline
import json


#json_file = "D:/A/Work/experiments/data/hate_dataset2_try.json"

# Load model
model_path = r'D:\A\Work\resources\twitter-roberta-base-sentiment-latest'
sentiment_pipeline = pipeline(
	"sentiment-analysis",
	model = model_path,
	tokenizer = model_path,
	truncation = True,
    max_length = 512
)

print("Loaded model and created pipeline")

# Input and output files
input_file = "D:/A/Work/experiments/data/hate_dataset3.json"
output_file = "D:/A/Work/experiments/data/hate_dataset3_sentiment.json"

batch_size = 512

results = []

# Load JSON data
with open(input_file, "r", encoding="utf-8") as f:
	data = json.load(f)

# Extract tweet texts
texts = [item["text"] for item in data]

# Process in batches
for i in range(0, len(texts), batch_size):
	if (i% batch_size == 0):
		print ("batch ", i)
	batch_texts = texts[i:i + batch_size]

	#sentiments = sentiment_pipeline(batch_texts)
	sentiments = sentiment_pipeline(batch_texts)

	for j, sentiment in enumerate(sentiments):
		item = data[i + j].copy()

		item["sentiment"] = sentiment["label"]
		item["sentiment_score"] = sentiment["score"]

		results.append(item)

# Save new JSON file
with open(output_file, "w", encoding="utf-8") as f:
	json.dump(results, f, indent = 4, ensure_ascii = False)

print("Sentiment analysis completed.")
print("Output saved to:", output_file)