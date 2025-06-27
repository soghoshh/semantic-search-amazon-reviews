import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def search(query, index_path, csv_path, top_k=5):
    print("[INFO] Loading model and data...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    df = pd.read_csv(csv_path)

    print("[INFO] Encoding query...")
    q_embed = model.encode([query]).astype('float32')

    print("[INFO] Loading index...")
    index = faiss.read_index(index_path)

    print("[INFO] Searching...")
    D, I = index.search(q_embed, top_k)

    print("\nTop results:")
    for idx in I[0]:
        print(f"- {df.iloc[idx]['Text']}\n")

if __name__ == "__main__":
    user_query = input("Enter your search query: ")
    search(user_query, 'embeddings/vector_store/index.faiss', 'embeddings/docs.csv')
