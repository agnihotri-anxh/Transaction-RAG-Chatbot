import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer

with open("transactions.json", "r") as f:
    transactions = json.load(f)

texts = [
    f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
    for t in transactions
]

with open("texts.json", "w") as f:
    json.dump(texts, f, indent=4)

model_path = "./models/paraphrase-MiniLM-L3-v2"
model = SentenceTransformer(model_path)

embeddings = model.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
).astype(np.float32)
embeddings = np.ascontiguousarray(embeddings, dtype=np.float16)

file_path = os.path.abspath("embeddings.npy")

if os.path.exists(file_path):
    os.remove(file_path)

np.save(file_path, embeddings, allow_pickle=False)

print("texts.json and embeddings.npy created successfully.")
