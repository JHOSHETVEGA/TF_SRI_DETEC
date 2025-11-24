import streamlit as st
import pandas as pd
import numpy as np

# Backend modules
from backend.model_training import train_truth_model
from backend.filtering import filter_real_news
from backend.indexing import build_tfidf_index, build_bm25_index
from backend.searching import search_tfidf, search_bm25
from backend.metrics import (
    precision_at_k, recall_at_k, average_precision, sri_confusion_matrix
)


# =========================================================
# TÍTULO
# =========================================================
st.title("📰 Sistema de Recuperación de Información con Noticias Reales (TF-IDF & BM25)")
st.write("Procesa tus datasets, detecta noticias reales, genera índices y prueba tu SRI con consultas.")


# =========================================================
# 1. SUBIDA DE ARCHIVOS
# =========================================================
st.header("1. Subir archivos CSV")

train_file = st.file_uploader("Train.csv", type="csv")
val_file   = st.file_uploader("Val.csv", type="csv")
test_file  = st.file_uploader("Test.csv", type="csv")

if not (train_file and val_file and test_file):
    st.info("Sube los tres archivos para continuar.")
    st.stop()

train = pd.read_csv(train_file)
val   = pd.read_csv(val_file)
test  = pd.read_csv(test_file)


# =========================================================
# 2. ENTRENAR MODELO REAL/FAKE
# =========================================================
st.header("2. Entrenar modelo de clasificación (REAL vs FAKE)")

if st.button("Entrenar modelo"):
    model, vectorizer, metrics = train_truth_model(train, val)

    st.success("Modelo entrenado correctamente.")
    st.write("### Accuracy")
    st.write(metrics["accuracy"])

    st.write("### Matriz de Confusión")
    st.write(metrics["confusion"])

    st.write("### Reporte de Clasificación")
    st.text(metrics["report"])

    st.session_state.model = model
    st.session_state.vectorizer = vectorizer


# Esperar modelo
if "model" not in st.session_state:
    st.warning("Primero entrena el modelo.")
    st.stop()


# =========================================================
# 3. FILTRAR NOTICIAS REALES
# =========================================================
st.header("3. Filtrar noticias reales del Test")

if st.button("Detectar noticias reales"):
    real_corpus = filter_real_news(
        st.session_state.model,
        st.session_state.vectorizer,
        test
    )

    st.success(f"Noticias reales detectadas: {len(real_corpus)}")
    st.dataframe(real_corpus.head())

    st.session_state.real_corpus = real_corpus


# Esperar corpus limpio
if "real_corpus" not in st.session_state:
    st.warning("Primero detecta noticias reales.")
    st.stop()


# =========================================================
# 4. CONSTRUIR ÍNDICES SRI
# =========================================================
st.header("4. Construir índices (TF-IDF y BM25)")

if st.button("Construir índices"):
    try:
        tfidf, tfidf_matrix = build_tfidf_index(st.session_state.real_corpus)
        bm25, tokens = build_bm25_index(st.session_state.real_corpus)

        st.session_state.tfidf = tfidf
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.bm25 = bm25

        st.success("Índices creados correctamente.")
    except Exception as e:
        st.error("Error al construir índices.")
        st.text(str(e))


if "tfidf" not in st.session_state or "bm25" not in st.session_state:
    st.warning("Primero construye los índices TF-IDF y BM25.")
    st.stop()


# =========================================================
# 5. BUSCAR EN EL SRI + MÉTRICAS + COMPARACIÓN FINAL
# =========================================================
st.header("5. Buscar en el Sistema de Recuperación de Información")

query = st.text_input("Escribe tu consulta:")

if st.button("Buscar"):
    real_corpus = st.session_state.real_corpus

    # ================================
    # TF-IDF
    # ================================
    results_tfidf, scores_tfidf = search_tfidf(
        query,
        st.session_state.tfidf,
        st.session_state.tfidf_matrix,
        real_corpus
    )

    st.subheader("Resultados TF-IDF")
    st.dataframe(results_tfidf)

    # ================================
    # BM25
    # ================================
    results_bm25, scores_bm25 = search_bm25(
        query,
        st.session_state.bm25,
        real_corpus
    )

    st.subheader("Resultados BM25")
    st.dataframe(results_bm25)

    # ================================
    # MÉTRICAS SRI
    # ================================
    st.header("📊 Métricas del SRI")

    relevance = np.ones(len(real_corpus))

    # ---- TF-IDF Metrics ----
    st.subheader("TF-IDF")
    st.write("Precision@5:", precision_at_k(scores_tfidf, relevance, k=5))
    st.write("Recall@5:", recall_at_k(scores_tfidf, relevance, k=5))
    st.write("Average Precision:", average_precision(scores_tfidf))
    cm_tfidf = sri_confusion_matrix(scores_tfidf, relevance)
    st.write("Matriz de Confusión (TF-IDF):")
    st.write(cm_tfidf)

    # ---- BM25 Metrics ----
    st.subheader("BM25")
    st.write("Precision@5:", precision_at_k(scores_bm25, relevance, k=5))
    st.write("Recall@5:", recall_at_k(scores_bm25, relevance, k=5))
    st.write("Average Precision:", average_precision(scores_bm25))
    cm_bm25 = sri_confusion_matrix(scores_bm25, relevance)
    st.write("Matriz de Confusión (BM25):")
    st.write(cm_bm25)

    # ================================
    # COMPARACIÓN FINAL
    # ================================
    st.header("📈 Comparación Final TF-IDF vs BM25")

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

    # Conclusión automática
    st.subheader("🧠 Conclusión Automática")

    p_diff = comparison_df.loc[0, "TF-IDF"] - comparison_df.loc[0, "BM25"]
    r_diff = comparison_df.loc[1, "TF-IDF"] - comparison_df.loc[1, "BM25"]
    ap_diff = comparison_df.loc[2, "TF-IDF"] - comparison_df.loc[2, "BM25"]

    conclusion = ""

    if p_diff > 0:
        conclusion += f"- TF-IDF tiene mayor Precision@5 (por {abs(p_diff):.4f})\n"
    else:
        conclusion += f"- BM25 tiene mayor Precision@5 (por {abs(p_diff):.4f})\n"

    if r_diff > 0:
        conclusion += f"- TF-IDF tiene mayor Recall@5 (por {abs(r_diff):.4f})\n"
    else:
        conclusion += f"- BM25 tiene mayor Recall@5 (por {abs(r_diff):.4f})\n"

    if ap_diff > 0:
        conclusion += f"- TF-IDF tiene mejor Average Precision.\n"
    else:
        conclusion += f"- BM25 tiene mejor Average Precision.\n"

    if (p_diff + r_diff + ap_diff) > 0:
        conclusion += "\n➡️ **TF-IDF es superior globalmente para**"
