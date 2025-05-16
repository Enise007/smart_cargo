# Başta sadece bir kez yapılır:
# - CSV okunur, embedding yapılır, FAISS kaydedilir

# Arayüz dosyası:
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import streamlit as st

# Yükleme
index = faiss.read_index("faiss_index.bin")
with open("questions.pkl", "rb") as f:
    questions = pickle.load(f)
df = pd.read_csv("questions_final_fixed_bom.csv")
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Arayüz
st.title("📦 Smart Cargo QA")
user_input = st.text_input("Type your question:")

if user_input:
    q_emb = model.encode([user_input])
    _, indices = index.search(np.array(q_emb), 1)
    idx = indices[0][0]

    st.subheader("Matched Question:")
    st.write(df.iloc[idx]["question"])
    st.subheader("Answer:")
    st.success(df.iloc[idx]["answer"])