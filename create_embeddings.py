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
embeddings = model.encode(texts, show_progress_bar=True).astype(np.float32)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")

try:
    # Use absolute path to avoid path issues
    import os
    file_path = os.path.abspath("embeddings.npy")
    print(f"Saving to: {file_path}")

    if os.path.exists(file_path):
        print("Removing old embeddings.npy file...")
        try:
            os.remove(file_path)
        except PermissionError:
            print("Warning: Could not remove old file. It might be in use.")
            print("Please close any programs using embeddings.npy and try again.")
            raise

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    
    # Save with allow_pickle=False for safety
    np.save(file_path, embeddings, allow_pickle=False)
    print("embeddings.npy created successfully!")

    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"File size: {file_size:.2f} MB")
        print(f"Embeddings saved: {embeddings.shape[0]} texts, {embeddings.shape[1]} dimensions")
    else:
        print("Warning: File was not created!")
        
except Exception as e:
    print(f"Error saving embeddings: {e}")
    print(f"Error type: {type(e)}")
    import traceback
    traceback.print_exc()
    raise
