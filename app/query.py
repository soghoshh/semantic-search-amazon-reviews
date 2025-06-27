import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

def load_index(index_path, docs_path):
    index = faiss.read_index(index_path)
    docs = pd.read_csv(docs_path)
    return index, docs

def semantic_search(query, index, docs, model, top_k=5):
    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype('float32'), top_k)
    results = docs.iloc[I[0]]
    return results
