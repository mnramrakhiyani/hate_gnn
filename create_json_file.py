# import csv
# import json

# csv_file_path = "D:/A/Work/experiments/data/hate_dataset.csv"
# json_file_path = "D:/A/Work/experiments/data/hate_dataset.json"

# data = []

# with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
	# csv_reader = csv.DictReader(csv_file)
	# #DictReader reads CSV as dictionaries (column names become keys)
	# for row in csv_reader:
		# data.append(row)

# with open(json_file_path, mode='w', encoding='utf-8') as json_file:
	# json.dump(data, json_file, indent=4)

# print("CSV successfully converted to JSON!")

# format of data should be id, text, label

# Code for first file
import pandas as pd

csv_file_path = "D:/A/Work/experiments/data/hate_dataset2.csv"
json_file_path = "D:/A/Work/experiments/data/hate_dataset2.json"


df = pd.read_csv(csv_file_path)

# Select 3rd, 4th, 5th columns
selected_df = df.iloc[:, 2:5]

selected_df.to_json(json_file_path, orient="records", indent=4)

print("Selected columns converted to JSON successfully!")


# Code for second file

import pandas as pd

# Read CSV
df = pd.read_csv(csv_file_path)

# Extract required columns
new_df = pd.DataFrame({
	"id": df.iloc[:, 0],		# First column
	"text": df.iloc[:, 6],	   # 7th column (index 6)
	"label": df.iloc[:, 5]	  # 6th column (index 5)
})

# Map numeric labels to text labels
label_mapping = {
	0: "hate",
	1: "offensive",
	2: "neither"
}

new_df["label"] = new_df["label"].map(label_mapping)

# Save as JSON
new_df.to_json(json_file_path, orient="records", indent=4)

print("JSON file created successfully!")