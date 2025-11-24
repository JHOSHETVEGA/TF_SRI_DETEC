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

st.title("📰 Sistema de Recuperación de Información con Noticias Verdaderas")


# --------------------------------------------
# 1. SUBIR DATASETS
# --------------------------------------------
st.header("1. Subir datasets")
train_file = st.file_uploader("Train.csv", type="csv")
val_file = st.file_uploader("Val.csv", type="csv")
test_file = st.file_uploader("Test.csv", type="csv")

if not (train_file and val_file and test_file):
    st.stop()

train = pd.read_csv(train_file)
val   = pd.read_csv(val_file)
test  = pd.read_csv(test_file)

mapping = {"FAKE":0, "REAL":1, "fake":0, "real":1}
for df in [train, val, test]:
    df["label"] = df["label"].map(mapping)


# --------------------------------------------
# 2. ENTRENAR MODELO INICIAL
# --------------------------------------------
st.header("2. Entrenar modelo para detectar noticias verdaderas")

if st.button("Entrenar modelo"):
    model, vectorizer, metrics = train_truth_model(train, val)

    st.subheader("Métricas del modelo inicial")
    st.write("Accuracy:", metrics["accuracy"])
    st.write(metrics["confusion"])
    st.text(metrics["report"])

    st.session_state["model"] = model
    st.session_state["vectorizer"] = vectorizer


# --------------------------------------------
# 3. FILTRAR NOTICIAS REALES DEL TEST
# --------------------------------------------
if "model" not in st.session_state:
    st.stop()

st.header("3. Filtrar noticias reales")

if st.button("Detectar noticias reales en Test"):
    real_corpus = filter_real_news(
        st.session_state["model"],
        st.session_state["vectorizer"],
        test
    )

    st.write("Noticias reales detectadas:", len(real_corpus))
    st.dataframe(real_corpus.head())

    st.session_state["real_corpus"] = real_corpus


# --------------------------------------------
# 4. CONSTRUIR ÍNDICES TF-IDF Y BM25
# --------------------------------------------
if "real_corpus" not in st.session_state:
    st.stop()

st.header("4. Construir índices SRI")

if st.button("Construir índices"):
    tfidf, tfidf_matrix = build_tfidf_index(st.session_state["real_corpus"])
    bm25, tokenized = build_bm25_index(st.session_state["real_corpus"])

    st.session_state["tfidf"] = tfidf
    st.session_state["tfidf_matrix"] = tfidf_matrix
    st.session_state["bm25"] = bm25

    st.success("Índices construidos correctamente.")


# --------------------------------------------
# 5. REALIZAR BÚSQUEDA + MÉTRICAS + COMPARACIÓN
# --------------------------------------------
if "tfidf" not in st.session_state:
    st.stop()

st.header("5. Buscar en el SRI")

query = st.text_input("Escribe tu consulta:")

if st.button("Buscar"):
    real_corpus = st.session_state["real_corpus"]

    # ----- TF-IDF -----
    query_vec = st.session_state["tfidf"].transform([query])
    scores = (st.session_state["tfidf_matrix"] @ query_vec.T).toarray().flatten()
    results_tfidf, scores_tfidf = search_tfidf(query, st.session_state["tfidf"], st.session_state["tfidf_matrix"], real_corpus)

    # ----- BM25 -----
    scores_bm = st.session_state["bm25"].get_scores(query.split())
    results_bm25, scores_bm25 = search_bm25(query, st.session_state["bm25"], real_corpus)

    # Mostrar resultados
    st.subheader("Resultados TF-IDF")
    st.write(results_tfidf)

    st.subheader("Resultados BM25")
    st.write(results_bm25)

    # ----- Métricas -----
    relevance = np.ones(len(real_corpus))

    st.subheader("Métricas TF-IDF")
    st.write("Precision@5:", precision_at_k(scores, relevance, k=5))
    st.write("Recall@5:", recall_at_k(scores, relevance, k=5))
    st.write("Average Precision:", average_precision(scores, relevance))
    cm_tfidf = sri_confusion_matrix(scores, relevance)
    st.write(cm_tfidf)

    st.subheader("Métricas BM25")
    st.write("Precision@5:", precision_at_k(scores_bm, relevance, k=5))
    st.write("Recall@5:", recall_at_k(scores_bm, relevance, k=5))
    st.write("Average Precision:", average_precision(scores_bm))
    cm_bm25 = sri_confusion_matrix(scores_bm, relevance)
    st.write(cm_bm25)

    # ----- Comparación Final -----
    st.header("📈 Comparación Final TF-IDF vs BM25")

    comparison_df = pd.DataFrame({
        "Métrica": [
            "Precision@5", "Recall@5", "Average Precision",
            "TP", "FP", "FN", "TN"
        ],
        "TF-IDF": [
            precision_at_k(scores, relevance, k=5),
            recall_at_k(scores, relevance, k=5),
            average_precision(scores, relevance),
            cm_tfidf[1, 1], cm_tfidf[0, 1], cm_tfidf[1, 0], cm_tfidf[0, 0]
        ],
        "BM25": [
            precision_at_k(scores_bm, relevance, k=5),
            recall_at_k(scores_bm, relevance, k=5),
            average_precision(scores_bm),
            cm_bm25[1, 1], cm_bm25[0, 1], cm_bm25[1, 0], cm_bm25[0, 0]
        ]
    })

    st.dataframe(comparison_df)
