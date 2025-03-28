import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# 모델 로딩
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# 저장된 청크 로딩
with open("chunked_texts.pkl", "rb") as f:
    chunked_texts = pickle.load(f)

# FAISS 인덱스 로딩
index = faiss.read_index("vector_db2.index")

# 검색 함수
def search_query(query, k=3):
    query_embedding = model.encode(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Streamlit UI
st.title("📄 문서 검색 시스템")

# 검색 입력
query = st.text_input("🔍 검색어를 입력하세요:")

if query:
    distances, indices = search_query(query)
    st.write("📌 검색 결과:")

    for i, idx in enumerate(indices[0]):
        st.write(f"**유사도:** {distances[0][i]:.4f}")
        st.write(f"**내용:** {chunked_texts[idx]}")
        st.write("---")

