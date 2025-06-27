# Semantic Search on Amazon Food Reviews

A Streamlit-powered semantic search app with FAISS vector store.

## Features
- SentenceTransformer embeddings
- FAISS vector index
- Streamlit UI for querying


## Getting Started
1. Clone repo
2. Install requirements
3. Run indexing script
4. Launch Streamlit UI

## Data
This repo includes a small sample dataset for demonstration.

To run with full data:
1. Download the full Amazon Food Reviews dataset from Kaggle.
2. Replace `data/reviews_sample.csv`.
3. Run `python ingestion.py` to build the FAISS index.

## Tech Stack
- Python
- FAISS
- SentenceTransformers
- Streamlit

## Future Improvements
- Advanced FAISS indexing
- Filtering in UI
- Cloud deployment

## License
MIT