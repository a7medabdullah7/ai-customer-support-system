import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

data = pd.read_csv("data/dataset-tickets-multi-lang3-4k.csv")

data["email_text"] = data["subject"] + " " + data["body"]
data = data.dropna(subset=["email_text", "answer"])

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(data["email_text"].tolist())

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "vector.index")

data.to_pickle("emails.pkl")

print("Vector database created successfully")