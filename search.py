import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
df = pd.read_csv("data/sentences_with_index.csv")
embeddings = np.load("embeddings.npy")

# Load model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def semantic_search(query, top_k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "text": df.iloc[idx]["text"],
            "language": df.iloc[idx]["language"],
            "score": similarities[idx]
        })
    return results

if __name__ == "__main__":
    query = input("Enter your query: ")
    results = semantic_search(query)

    print("\nTop results:")
    for r in results:
        print(f"{r['text']} ({r['language']}) -> score: {r['score']:.3f}")
