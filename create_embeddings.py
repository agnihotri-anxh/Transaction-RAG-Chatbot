import json
import numpy as np
from sentence_transformers import SentenceTransformer

with open("transactions.json", "r") as f:
    transactions = json.load(f)

texts = [
    f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
    for t in transactions
]

with open("texts.json", "w") as f:
    json.dump(texts, f, indent=4)

print("texts.json created successfully!")

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

np.save("embeddings.npy", embeddings)

print("embeddings.npy created successfully!")
