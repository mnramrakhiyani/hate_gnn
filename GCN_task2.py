# Task2 : edge creation using embending using sentiments similarity 
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

import torch.nn.functional as F
from torch_geometric.nn import GCNConv


json_file = "D:/A/Work/experiments/data/hate_dataset1_sentiment.json"


class GCN(torch.nn.Module):
	def __init__(self, input_dim, hidden_dim, num_classes):
		super(GCN, self).__init__()
		self.conv1 = GCNConv(input_dim, hidden_dim)
		self.conv2 = GCNConv(hidden_dim, num_classes)

	def forward(self, data):
		x, edge_index = data.x, data.edge_index
		
		x = self.conv1(x, edge_index)
		x = F.relu(x)
		x = self.conv2(x, edge_index)
		return x

seed = 42

# Load JSON
df = pd.read_json(json_file)
#df_sample = df.sample(2000, random_state=42)
df_sample = df.sample(2000, random_state = seed).reset_index(drop=True)
#df_sample = df.reset_index(drop=True)

torch.manual_seed(seed)

# Load pretrained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to embeddings
embeddings = model.encode(df_sample['text'].tolist(), batch_size=32,show_progress_bar=True)

# Compute similarity matrix
sim_matrix = cosine_similarity(embeddings)

G = nx.Graph()

# Add nodes
# for i in range(len(df_sample)):
	# G.add_node(i, label=df_sample.iloc[i]['label'])

for i, row in df_sample.iterrows():
	#G.add_node(row['id'], label=row['label'])
	# dataset1 instead of id it is X1
	G.add_node(row['X1'], label=row['sentiment'])
# Add edges
threshold = 0.5

for i in range(len(df_sample)):
	for j in range(i+1, len(df_sample)):
		if sim_matrix[i][j] > threshold:
			G.add_edge(i, j)

edge_list = list(G.edges())
# converts it into a PyTorch tensor of integers. and .t() transposes it. .contiguous() → ensures memory layout is continuous (required after transpose).
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Make graph undirected:
edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

# Node Features (Embeddings)
x = torch.tensor(embeddings, dtype=torch.float)

# Encode labels
le = LabelEncoder()
#labels = le.fit_transform(df_sample['label'])
labels = le.fit_transform(df_sample['sentiment'])

# Converts encoded labels to tensor.
y = torch.tensor(labels, dtype=torch.long)

# Create data object 

data = Data(x=x, edge_index=edge_index, y=y)

# Create Train/Test Split
num_nodes = data.num_nodes		# To get number of nodes in the dataset
indices = torch.randperm(num_nodes)

train_size = int(0.8 * num_nodes)

train_idx = indices[:train_size]
test_idx = indices[train_size:]

data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

data.train_mask[train_idx] = True
data.test_mask[test_idx] = True

# train model
model = GCN(
	input_dim=data.num_features,
	hidden_dim=64,
	num_classes=len(le.classes_)
)

print ("trained model")

optimizer = torch.optim.Adam(model. parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(200):
	model.train()
	optimizer.zero_grad()
	
	out = model(data)
	loss = criterion(out[data.train_mask], data.y[data.train_mask])
	#acc = (out[data.train_mask] == data.y[data.train_mask]).sum()/ int(data.y[data.train_mask].sum())
	
	loss.backward()
	optimizer.step()
	
	if epoch % 10 == 0:
		print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# evaluate
model.eval()
out = model(data)
pred = out.argmax(dim=1)

# Select test nodes
y_true = data.y[data.test_mask]
y_pred = pred[data.test_mask]

# Confusion matrix components (binary classification)
TP = ((y_pred == 1) & (y_true == 1)).sum().item()
TN = ((y_pred == 0) & (y_true == 0)).sum().item()
FP = ((y_pred == 1) & (y_true == 0)).sum().item()
FN = ((y_pred == 0) & (y_true == 1)).sum().item()

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