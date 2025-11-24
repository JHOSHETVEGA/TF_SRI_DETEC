import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Backend modules
from backend.model_training import train_truth_model
from backend.indexing import build_tfidf_index, build_bm25_index
from backend.searching import search_tfidf, search_bm25
from backend.metrics import (
    precision_at_k, recall_at_k, average_precision
)

# =========================================================
# 4×4 MATRIX FUNCTION
# =========================================================
def sri_confusion_4way(scores, true_labels, threshold=0.0):
    """
    true_labels = 1 si es REAL, 0 si es FAKE
    scores = puntaje TF-IDF o BM25
    threshold = 0 (score>0 = relevante)
    """

    predicted_relevance = (scores > threshold).astype(int)

    # 4 categorías SRI
    TP_real = np.sum((true_labels == 1) & (predicted_relevance == 1))
    FN_real = np.sum((true_labels == 1) & (predicted_relevance == 0))

    FP_fake = np.sum((true_labels == 0) & (predicted_relevance == 1))
    TN_fake = np.sum((true_labels == 0) & (predicted_relevance == 0))

    return pd.DataFrame({
        "Pred. Relevante":   [TP_real, FP_fake],
        "Pred. Irrelevante": [FN_real, TN_fake]
    }, index=["Real", "Fake"])


# =========================================================
# TÍTULO
# =========================================================
st.title("📰 Clasificación REAL/FAKE + SRI (TF-IDF & BM25) con Matriz 4×4")
st.write("Entrena un clasificador, predice en Test, construye un SRI y evalúa con matriz 4×4.")

# =========================================================
# 1. SUBIDA DE ARCHIVOS
# =========================================================
st.header("1. Subir archivos Train.csv, Val.csv y Test.csv")

train_file = st.file_uploader("Train.csv", type="csv")
val_file   = st.file_uploader("Val.csv", type="csv")
test_file  = st.file_uploader("Test.csv", type="csv")

if not (train_file and val_file and test_file):
    st.info("Sube los tres archivos para continuar.")
    st.stop()

train = pd.read_csv(train_file)
val   = pd.read_csv(val_file)
test  = pd.read_csv(test_file)

train["tweet"] = train["tweet"].astype(str)
val["tweet"]   = val["tweet"].astype(str)
test["tweet"]  = test["tweet"].astype(str)

st.success("Archivos cargados correctamente.")


# =========================================================
# 2. ENTRENAR MODELO REAL/FAKE
# =========================================================
st.header("2. Entrenar modelo REAL vs FAKE")

if st.button("Entrenar modelo"):

    model, vectorizer, metrics = train_truth_model(train, val)

    st.success("Modelo entrenado correctamente.")

    st.write("### Accuracy (Validación)")
    st.write(metrics["accuracy"])

    st.write("### Matriz de Confusión (Validación)")
    st.write(metrics["confusion"])

    st.write("### Reporte de Clasificación (Validación)")
    st.text(metrics["report"])

    st.session_state.model = model
    st.session_state.vectorizer = vectorizer

    # PREDICCIÓN EN TEST
    X_test = vectorizer.transform(test["tweet"])
    preds_test = model.predict(X_test)

    test_pred = test.copy()
    test_pred["pred_label"] = preds_test
    test_pred["pred_label_text"] = test_pred["pred_label"].map({0: "fake", 1: "real"})

    st.write("### Predicciones en Test.csv")
    st.dataframe(test_pred.head(20))

    st.session_state.test_df = test_pred


if "test_df" not in st.session_state:
    st.warning("Entrena el modelo para continuar.")
    st.stop()



# =========================================================
# 3. PREPARAR CORPUS COMPLETO PARA SRI
# =========================================================
st.header("3. Preparar corpus completo del SRI")

corpus_df = st.session_state.test_df.copy().reset_index(drop=True)
relevance = corpus_df["pred_label"].values.astype(int)  # REAL = relevante

st.success(f"Corpus SRI listo: {len(corpus_df)} documentos")
st.write(f"Relevantes (REAL) según clasificador: {relevance.sum()} / {len(relevance)}")

st.session_state.corpus_df = corpus_df
st.session_state.relevance = relevance


# =========================================================
# 4. CONSTRUIR TF-IDF Y BM25
# =========================================================
st.header("4. Construir índices TF-IDF y BM25")

if st.button("Construir índices"):
    try:
        tfidf, tfidf_matrix = build_tfidf_index(corpus_df)
        bm25, tokens        = build_bm25_index(corpus_df)

        st.session_state.tfidf = tfidf
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.bm25 = bm25

        st.success("Índices creados correctamente.")
    except Exception as e:
        st.error("Error construyendo índices.")
        st.text(str(e))


if "tfidf" not in st.session_state:
    st.warning("Primero construye los índices.")
    st.stop()



# =========================================================
# 5. BÚSQUEDA + TOP-5 + MATRIZ 4×4
# =========================================================
st.header("5. Buscar en el SRI (TF-IDF y BM25)")

query = st.text_input("Escribe tu consulta:")

if st.button("Buscar"):

    corpus_df = st.session_state.corpus_df
    relevance = st.session_state.relevance

    # ---------------- TF-IDF TOP-5 ----------------
    results_tfidf, scores_top_tfidf = search_tfidf(
        query,
        st.session_state.tfidf,
        st.session_state.tfidf_matrix,
        corpus_df,
        top_k=5
    )

    results_tfidf["score"] = scores_top_tfidf

    st.subheader(" Top-5 TF-IDF ")
    st.dataframe(results_tfidf)

    # ---------------- BM25 TOP-5 ----------------
    results_bm25, scores_top_bm25 = search_bm25(
        query,
        st.session_state.bm25,
        corpus_df,
        top_k=5
    )

    results_bm25["score"] = scores_top_bm25

    st.subheader(" Top-5 BM25 ")
    st.dataframe(results_bm25)

    # =====================================================
    # MÉTRICAS DEL SRI
    # =====================================================
    st.header(" Métricas del SRI")

    # Puntajes completos TF-IDF
    query_vec = st.session_state.tfidf.transform([query])
    all_scores_tfidf = (st.session_state.tfidf_matrix @ query_vec.T).toarray().flatten()

    # Puntajes completos BM25
    query_tokens = query.split()
    all_scores_bm25 = st.session_state.bm25.get_scores(query_tokens)

    # ----------- TF-IDF -----------
    st.subheader("TF-IDF")
    st.write("Precision@5:", precision_at_k(all_scores_tfidf, relevance, 5))
    st.write("Recall@5:",    recall_at_k(all_scores_tfidf, relevance, 5))
    st.write("Average Precision:", average_precision(all_scores_tfidf, relevance))

    # ----------- BM25 -----------
    st.subheader("BM25")
    st.write("Precision@5:", precision_at_k(all_scores_bm25, relevance, 5))
    st.write("Recall@5:",    recall_at_k(all_scores_bm25, relevance, 5))
    st.write("Average Precision:", average_precision(all_scores_bm25, relevance))

    # =====================================================
    # MATRIZ 4×4 REAL/FAKE × RELEVANTE/IRRELEVANTE
    # =====================================================
    st.header(" Matriz 4×4 del SRI")

    st.subheader("Matriz TF-IDF (4×4)")
    cm4_tfidf = sri_confusion_4way(all_scores_tfidf, relevance)
    st.write(cm4_tfidf)

    st.subheader("Matriz BM25 (4×4)")
    cm4_bm25 = sri_confusion_4way(all_scores_bm25, relevance)
    st.write(cm4_bm25)


    # =====================================================
    # COMPARACIÓN FINAL
    # =====================================================
    st.header("📈 Comparación Final TF-IDF vs BM25")

    tfidf_score_total = (
        precision_at_k(all_scores_tfidf, relevance, 5) +
        recall_at_k(all_scores_tfidf, relevance, 5) +
        average_precision(all_scores_tfidf, relevance)
    )

    bm25_score_total = (
        precision_at_k(all_scores_bm25, relevance, 5) +
        recall_at_k(all_scores_bm25, relevance, 5) +
        average_precision(all_scores_bm25, relevance)
    )

    if tfidf_score_total > bm25_score_total:
        st.success("➡️ **TF-IDF supera a BM25 para esta consulta.**")
    elif bm25_score_total > tfidf_score_total:
        st.success("➡️ **BM25 supera a TF-IDF para esta consulta.**")
    else:
        st.info("➡️ **TF-IDF y BM25 tienen desempeño equivalente.**")
