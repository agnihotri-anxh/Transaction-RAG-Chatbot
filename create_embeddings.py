import json
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

with open("transactions.json", "r") as f:
    transactions = json.load(f)

texts = [
    f"On {t['date']}, {t['customer']} purchased a {t['product']} for ₹{t['amount']}."
    for t in transactions
]

with open("texts.json", "w") as f:
    json.dump(texts, f, indent=4)

print("✔ Saved texts.json")

print("Loading model...")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")


print("Encoding embeddings...")

emb_list = []

for i in tqdm(range(0, len(texts), 64)):
    batch = texts[i:i+64]
    emb = model.encode(
        batch,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float16)
    emb_list.append(emb)

embeddings = np.vstack(emb_list)


save_path = "embeddings_fp16.npy"
if os.path.exists(save_path):
    os.remove(save_path)

np.save(save_path, embeddings, allow_pickle=False)

print("✔ embeddings_fp16.npy created successfully")

