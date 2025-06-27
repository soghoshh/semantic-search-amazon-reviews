import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

def create_index(csv_path, index_path):
    print("[INFO] Loading reviews...")
    df = pd.read_csv(csv_path)
    texts = df['Text'].dropna().tolist()

    print(f"[INFO] {len(texts)} reviews loaded.")

    print("[INFO] Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("[INFO] Generating embeddings...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    dim = embeddings.shape[1]

    print("[INFO] Building FAISS index...")
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))

    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    df.to_csv('embeddings/docs.csv', index=False)

    print("[SUCCESS] Index and CSV saved.")

if __name__ == "__main__":
    create_index('data/reviews_sample.csv', 'embeddings/vector_store/index.faiss')
