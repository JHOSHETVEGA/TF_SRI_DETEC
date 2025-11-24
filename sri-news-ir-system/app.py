import streamlit as st
import pandas as pd
import numpy as np

from backend.model_training import train_truth_model
from backend.filtering import filter_real_news
from backend.indexing import build_tfidf_index, build_bm25_index
from backend.searching import search_tfidf, search_bm25
from backend.metrics import (
    precision_at_k, recall_at_k, average_precision, sri_confusion_matrix
)

st.title("📰 Sistema de Recuperación de Información con Noticias Reales")


# =========================================================
# 1. Subida de archivos
# =========================================================
train_file = st.file_uploader("Sube Train.csv", type="csv")
val_file   = st.file_uploader("Sube Val.csv", type="csv")
test_file  = st.file_uploader("Sube Test.csv", type="csv")

if not (train_file and val_file and test_file):
    st.stop()

train = pd.read_csv(train_file)
val   = pd.read_csv(val_file)
test  = pd.read_csv(test_file)


# =========================================================
# 2. Entrenar modelo
# =========================================================
if st.button("Entrenar modelo"):
    model, vectorizer, metrics = train_truth_model(train, val)

    st.write("Accuracy:", metrics["accuracy"])
    st.write(metrics["confusion"])
    st.text(metrics["report"])

    st.session_state["model"] = model
    st.session_state["vectorizer"] = vectorizer


# =========================================================
# 3. Filtrar noticias reales
# =========================================================
if "model" not in st.session_state:
    st.stop()

if st.button("Filtrar noticias reales del Test"):
    real_corpus = filter_real_news(
        st.session_state["model"],
        st.session_state["vectorizer"],
        test
    )

    st.write("Noticias reales detectadas:", len(real_corpus))
    st.dataframe(real_corpus.head())

    st.session_state["real_corpus"] = real_corpus


# =========================================================
# 4. Construir índices SRI
# =========================================================
if "real_corpus" not in st.session_state:
    st.stop()

if st.button("Construir índices TF-IDF y BM25"):
    tfidf, tfidf_matrix = build_tfidf_index(st.session_state["real_corpus"])
    bm25, tokenized      = build_bm25_index(st.session_state["real_corpus"])

    st.session_state["tfidf"] = tfidf
    st.session_state["tfidf_matrix"] = tfidf_matrix
    st.session_state["bm25"] = bm25

    st.success("Índices creados correctamente.")


# =========================================================
# 5. Búsqueda + Métricas + Comparación
# =========================================================
query = st.text_input("Escribe una consulta:")

if st.button("Buscar"):
    real_corpus = st.session_state["real_corpus"]

    # TF-IDF
    results_tfidf, scores_tfidf = search_tfidf(
        query,
        st.session_state["tfidf"],
        st.session_state["tfidf_matrix"],
        real_corpus
    )

    # BM25
    results_bm25, scores_bm25 = search_bm25(
        query,
        st.session_state["bm25"],
        real_corpus
    )

    # Mostrar resultados
    st.subheader("Resultados TF-IDF")
    st.write(results_tfidf)

    st.subheader("Resultados BM25")
    st.write(results_bm25)

    # Métricas del SRI
    relevance = np.ones(len(real_corpus))

    st.header("📊 Métricas del SRI")

    st.write("Precision@5 (TF-IDF):", precision_at_k(scores_tfidf, relevance))
    st.write("Recall@5 (TF-IDF):", recall_at_k(scores_tfidf, relevance))
    st.write("Average Precision (TF-IDF):", average_precision(scores_tfidf))
    cm_tfidf = sri_confusion_matrix(scores_tfidf, relevance)
    st.write(cm_tfidf)

    st.write("Precision@5 (BM25):", precision_at_k(scores_bm25, relevance))
    st.write("Recall@5 (BM25):", recall_at_k(scores_bm25, relevance))
    st.write("Average Precision (BM25):", average_precision(scores_bm25))
    cm_bm25 = sri_confusion_matrix(scores_bm25, relevance)
    st.write(cm_bm25)

    # COMPARACIÓN FINAL
    st.header("📈 Comparación Final")

    comparison_df = pd.DataFrame({
        "Métrica": ["Precision@5", "Recall@5", "Average Precision"],
        "TF-IDF": [
            precision_at_k(scores_tfidf, relevance),
            recall_at_k(scores_tfidf, relevance),
            average_precision(scores_tfidf)
        ],
        "BM25": [
            precision_at_k(scores_bm25, relevance),
            recall_at_k(scores_bm25, relevance),
            average_precision(scores_bm25)
        ]
    })

    st.dataframe(comparison_df)
