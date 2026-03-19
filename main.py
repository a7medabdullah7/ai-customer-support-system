import pandas as pd
import ollama

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ===============================
# Load Dataset
# ===============================
def load_dataset():

    print("Loading dataset...")

    data = pd.read_csv("data/dataset-tickets-multi-lang3-4k.csv")

    data["email_text"] = data["subject"] + " " + data["body"]

    data = data.dropna(subset=["email_text", "answer"])

    print("Dataset size:", len(data))

    return data


# ===============================
# Load Embedding Model
# ===============================
def load_model():

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    return model


# ===============================
# Create Embeddings
# ===============================
def create_embeddings(model, data):

    print("Creating embeddings...")

    embeddings = model.encode(data["email_text"].tolist())

    print("Embeddings ready")

    return embeddings


# ===============================
# Semantic Search
# ===============================
def search_similar_email(query, model, embeddings, data):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(query_embedding, embeddings)[0]

    best_match_index = similarities.argmax()

    solution = data.iloc[best_match_index]["answer"]

    score = similarities[best_match_index]

    return solution, score


# ===============================
# Generate AI Reply using LLaMA
# ===============================
def generate_reply(query, solution):

    prompt = f"""
Customer email:
{query}

Support solution:
{solution}

Write a professional customer support email reply.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    reply = response["message"]["content"]

    return reply


# ===============================
# Main Program
# ===============================
def main():

    data = load_dataset()

    model = load_model()

    embeddings = create_embeddings(model, data)

    query = input("\nEnter customer email:\n")

    solution, score = search_similar_email(query, model, embeddings, data)

    print("\nSimilarity Score:", score)

    reply = generate_reply(query, solution)

    print("\nAI Reply:\n")

    print(reply)


# ===============================
# Run Program
# ===============================
if __name__ == "__main__":

    main()