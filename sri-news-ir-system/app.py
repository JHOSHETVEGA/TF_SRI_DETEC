import streamlit as st
import pandas as pd
import numpy as np

from backend.indexing import build_tfidf_index, build_bm25_index
from backend.searching import search_tfidf, search_bm25
from backend.metrics import (
    precision_at_k, recall_at_k, average_precision, sri_confusion_matrix
)

from sentence_transformers import SentenceTransformer, util


# =========================================================
# TÍTULO
# =========================================================
st.title("📰 Sistema de Recuperación de Información (TF-IDF vs BM25)")
st.write("SRI completo utilizando TODO el corpus, sin filtrar noticias verdaderas.")


# =========================================================
# 1. SUBIR ARCHIVOS
# =========================================================
st.header("1. Subir archivo Test.csv")

test_file = st.file_uploader("Sube tu Test.csv aquí:", type="csv")

if not test_file:
    st.info("Sube el archivo Test.csv para continuar.")
    st.stop()

test = pd.read_csv(test_file)
corpus = test["tweet"].tolist()

st.success(f"Corpus cargado correctamente: {len(corpus)} documentos.")


# =========================================================
# 2. CONSTRUIR ÍNDICES COMPLETOS
# =========================================================
st.header("2. Construir índices (TF-IDF y BM25)")

if st.button("Construir índices"):
    try:
        tfidf, tfidf_matrix = build_tfidf_index(corpus)
        bm25, tokenized = build_bm25_index(corpus)

        st.session_state.tfidf = tfidf
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.bm25 = bm25
        st.session_state.corpus = corpus

        st.success("Índices construidos correctamente.")

    except Exception as e:
        st.error("Error construyendo los índices.")
        st.text(str(e))


if "tfidf" not in st.session_state:
    st.warning("Primero construye los índices.")
    st.stop()


# =========================================================
# 3. RELEVANCIA AUTOMÁTICA CON EMBEDDINGS
# =========================================================
st.header("3. Buscar con SRI + Relevancia Automática")

query = st.text_input("Escribe tu consulta:")

if st.button("Buscar"):
    corpus = st.session_state.corpus

    # ======================================
    # A) RANKING TF-IDF
    # ======================================
    results_tfidf, scores_top_tfidf = search_tfidf(
        query,
        st.session_state.tfidf,
        st.session_state.tfidf_matrix,
        corpus
    )

    st.subheader("Resultados TF-IDF")
    st.dataframe(results_tfidf)


    # ======================================
    # B) RANKING BM25
    # ======================================
    results_bm25, scores_top_bm25 = search_bm25(
        query,
        st.session_state.bm25,
        corpus
    )

    st.subheader("Resultados BM25")
    st.dataframe(results_bm25)


    # ====================================================
    # 4. GENERAR RELEVANCIA AUTOMÁTICA (GROUND TRUTH)
    # ====================================================
    st.subheader("Generando relevancia automática...")

    embedder = SentenceTransformer("all-mpnet-base-v2")

    query_emb = embedder.encode(query)
    doc_emb = embedder.encode(corpus)

    # similaridad semántica para evaluar relevancia real
    semantic_scores = util.cos_sim(query_emb, doc_emb)[0].cpu().numpy()

    # documento relevante si su similaridad > threshold
    threshold = 0.40
    relevance = (semantic_scores >= threshold).astype(int)

    st.success(f"Documentos relevantes detectados: {relevance.sum()} / {len(relevance)}")


    # ====================================================
    # 5. MÉTRICAS SRI (AHORA SI REALES)
    # ====================================================
    st.header("📊 Métricas del SRI")

    # Scores completos para ranking
    all_scores_tfidf = (st.session_state.tfidf_matrix @ st.session_state.tfidf.transform([query]).T).toarray().flatten()
    all_scores_bm25 = st.session_state.bm25.get_scores(query.split())

    # ----- TF-IDF -----
    st.subheader("TF-IDF")
    st.write("Precision@5:", precision_at_k(score
