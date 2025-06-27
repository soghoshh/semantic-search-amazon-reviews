import streamlit as st
import pandas as pd
from query import load_index, semantic_search
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(page_title="Semantic Search", layout="centered")

# App title
st.title("Amazon Food Reviews Semantic Search")

# Load FAISS index and data
index, docs = load_index('embeddings/vector_store/index.faiss', 'embeddings/docs.csv')
docs_df = pd.read_csv('embeddings/docs.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

# User input
query = st.text_input("Enter your search query:")

if st.button("Search") and query:
    with st.spinner("Searching..."):
        results = semantic_search(query, index, docs_df, model=model)
        if results.empty:
            st.warning("No results found. Try another query!")
        else:
            st.success(f"Top {len(results)} results:")
            for i, row in results.iterrows():
                st.markdown(f"**Score:** {row['Score']:.4f}")
                st.markdown(f"{row['Text']}")
                st.markdown("---")
