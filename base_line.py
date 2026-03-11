from transformers import pipeline
import torch
import json
import pandas as pd

#json_file = "D:/A/Work/experiments/data/hate_dataset2_try.json"

# # Load model
# model_path = r'D:\A\Work\resources\twitter-roberta-base-sentiment-latest'
# sentiment_pipeline = pipeline(
	# "sentiment-analysis",
	# model = model_path,
	# tokenizer = model_path,
	# truncation = True,
    # max_length = 512
# )

# print("Loaded model and created pipeline")

# Input and output files
json_file = "D:/A/Work/experiments/data/hate_dataset1_sentiment.json"

# output_file = "D:/A/Work/experiments/data/hate_dataset3_sentiment.json"

batch_size = 512

results = []

# Load JSON data
#with open(input_file, "r", encoding="utf-8") as f:
#	data = json.load(f)
#print ("Total tweets:", str(len(data)))

seed = 42

# Load JSON
df = pd.read_json(json_file)
#df_sample = df.sample(2000, random_state=42)
df_sample = df.sample(2000, random_state = seed).reset_index(drop=True)
#df_sample = df.reset_index(drop=True)

torch.manual_seed(seed)

# Create Train/Test Split
num_nodes = len(df_sample['text'])		# To get number of nodes in the dataset
indices = torch.randperm(num_nodes)

train_size = int(0.8 * num_nodes)

train_idx = indices[:train_size]
test_idx = indices[train_size:]

TP = 0
FN = 0
FP = 0 
TN = 0
count = 0
y_pred = []
for idx in test_idx:
	labels = df_sample["label"][idx.item()]
	sentiments = df_sample["sentiment"][idx.item()]
	sent_score = df_sample["sentiment_score"][idx.item()]

	if sent_score >= 0.6:
		count += 1
		#DS1
		if labels == "hate" and sentiments == "negative":
			TP += 1
		elif labels == "hate" and sentiments == "positive":
			FN += 1
		elif labels == "nothate" and sentiments == "negative":
			FP += 1
		elif labels == "nothate" and sentiments == "positive":
			TN += 1
		
		#DS2
		"""if (labels == "hate" or labels == "offensive") and sentiments == "negative":
			TP += 1
		elif (labels == "hate" or labels == "offensive") and (sentiments == "positive" or sentiments == "neutral"):
			FN += 1
		elif labels == "neither" and (sentiments == "negative"):
			FP += 1
		elif labels == "neither" and (sentiments == "positive" or sentiments == "neutral"):
			TN += 1
		#DS3
		if (labels == "hate" or labels == "offensive") and sentiments == "negative":
			TP += 1
		elif (labels == "hate" or labels == "offensive") and (sentiments == "positive" or sentiments == "neutral"):
			FN += 1
		elif labels == "normal" and (sentiments == "negative"):
			FP += 1
		elif labels == "normal" and (sentiments == "positive" or sentiments == "neutral"):
			TN += 1"""

print ("Total tweets we considered:", count)
# Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Precision
precision = TP / (TP + FP) if (TP + FP) > 0 else 0

# Recall (Sensitivity)
recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# F1 Score
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Specificity
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("Specificity:", specificity)

# print (data.head())
# # Extract tweet texts
# labels, sentiments,sent_score = [(item["label"], item["sentiment"], item["senti_score"]) for item in data]

#print (str(len(labels)) + "\t" + str(len(sentiments)))



# # Process in batches
# for i in range(0, len(labels), batch_size):
	# if (i% batch_size == 0):
		# print ("batch ", i)


	# batch_texts = texts[i:i + batch_size]

	# #sentiments = sentiment_pipeline(batch_texts)
	# sentiments = sentiment_pipeline(batch_texts)

	# for j, sentiment in enumerate(sentiments):
		# item = data[i + j].copy()

		# item["sentiment"] = sentiment["label"]
		# item["sentiment_score"] = sentiment["score"]

		# results.append(item)

# # Save new JSON file
# with open(output_file, "w", encoding="utf-8") as f:
	# json.dump(results, f, indent = 4, ensure_ascii = False)

# print("Sentiment analysis completed.")
# print("Output saved to:", output_file)