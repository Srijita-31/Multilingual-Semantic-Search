import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("data/sentences.csv")

# Load multilingual model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Generate embeddings
embeddings = model.encode(df["text"].tolist())

# Save embeddings
np.save("embeddings.npy", embeddings)
df.to_csv("data/sentences_with_index.csv", index=False)

print("Embeddings generated successfully!")
