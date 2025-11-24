import streamlit as st
import pandas as pd
import numpy as np

# Backend modules
from backend.model_training import train_truth_model
from backend.indexing import build_tfidf_index, build_bm25_index
from backend.searching import search_tfidf, search_bm25
from backend.metrics import (
    precision_at_k, recall_at_k, average_precision, sri_confusion_matrix
)

# =========================================================
# TÍTULO
# =========================================================
st.title("📰 Clasificación REAL/FAKE + Sistema de Recuperación de Información (TF-IDF & BM25)")
st.write(
    "Aplicación que primero entrena un modelo para clasificar noticias como reales o falsas "
    "y luego construye un Sistema de Recuperación de Información (SRI) sobre TODO el Test.csv "
    "usando TF-IDF y BM25."
)

# =========================================================
# 1. SUBIDA DE ARCHIVOS
# =========================================================
st.header("1. Subir archivos CSV (Train, Val, Test)")

train_file = st.file_uploader("Train.csv", type="csv")
val_file   = st.file_uploader("Val.csv", type="csv")
test_file  = st.file_uploader("Test.csv", type="csv")

if not (train_file and val_file and test_file):
    st.info("Sube los tres archivos para continuar.")
    st.stop()

train = pd.read_csv(train_file)
val   = pd.read_csv(val_file)
test  = pd.read_csv(test_file)

# Aseguramos tipos correctos
train["tweet"] = train["tweet"].astype(str)
val["tweet"]   = val["tweet"].astype(str)
test["tweet"]  = test["tweet"].astype(str)

# Mapeo de etiquetas para TODO el dataset (0 = fake, 1 = real)
label_mapping = {"fake": 0, "real": 1}
for df in (train, val):
    df["label_bin"] = df["label"].str.lower().map(label_mapping)
    df["label_bin"] = df["label_bin"].fillna(0).astype(int)

st.success("Archivos cargados y etiquetas mapeadas (fake=0, real=1).")


# =========================================================
# 2. ENTRENAR MODELO REAL/FAKE
# =========================================================
st.header("2. Entrenar modelo de clasificación (REAL vs FAKE)")

if st.button("Entrenar modelo Naive Bayes"):
    # Usamos la función modular que limpia y entrena
    model, vectorizer, metrics = train_truth_model(train, val)

    st.success("✅ Modelo entrenado correctamente sobre Train/Val.")
    st.write("### Accuracy (Validación)")
    st.write(metrics["accuracy"])

    st.write("### Matriz de Confusión (Validación)")
    st.write(metrics["confusion"])

    st.write("### Reporte de Clasificación (Validación)")
    st.text(metrics["report"])

    # Guardamos en sesión
    st.session_state.model = model
    st.session_state.vectorizer = vectorizer

    # Evaluación adicional en Test
    X_test = vectorizer.transform(test["tweet"])
    preds_test = model.predict(X_test)

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    acc_test = accuracy_score(test["label_bin"], preds_test)
    cm_test  = confusion_matrix(test["label_bin"], preds_test)
    rep_test = classification_report(test["label_bin"], preds_test)

    st.write("### Accuracy (Test)")
    st.write(acc_test)

    st.write("### Matriz de Confusión (Test)")
    st.write(cm_test)

    st.write("### Reporte de Clasificación (Test)")
    st.text(rep_test)


if "model" not in st.session_state:
    st.warning("Entrena primero el modelo para continuar (sección 2).")
else:
    st.info("Modelo cargado en memoria. Puedes continuar con el SRI.")

# =========================================================
# 3. PREPARAR CORPUS COMPLETO DEL SRI
# =========================================================
st.header("3. Preparar el corpus para el SRI (sin filtrar)")

# Usamos TODO el Test.csv como corpus de documentos del SRI
corpus_df = test.copy()   # DataFrame completo
corpus_df = corpus_df.reset_index(drop=True)

st.write(f"El corpus del SRI contiene **{len(corpus_df)}** documentos (todas las noticias del Test).")
st.session_state.corpus_df = corpus_df

# Relevancia para el SRI:
# Definimos que una noticia es relevante si es REAL (label_bin = 1)
relevance = corpus_df["label_bin"].values
st.write(f"Noticias REAL (relevantes) en el corpus: {relevance.sum()} / {len(relevance)}")


# =========================================================
# 4. CONSTRUIR ÍNDICES TF-IDF Y BM25 SOBRE TODO EL TEST
# =========================================================
st.header("4. Construir índices de Recuperación de Información (TF-IDF y BM25)")

if st.button("Construir índices SRI"):
    try:
        tfidf, tfidf_matrix = build_tfidf_index(corpus_df)
        bm25, tokens        = build_bm25_index(corpus_df)

        st.session_state.tfidf = tfidf
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.bm25 = bm25

        st.success("✅ Índices TF-IDF y BM25 construidos correctamente sobre TODO el Test.csv.")
    except Exception as e:
        st.error("Error al construir índices.")
        st.text(str(e))

if "tfidf" not in st.session_state or "bm25" not in st.session_state:
    st.warning("Construye primero los índices del SRI (sección 4) para poder buscar.")
    st.stop()


# =========================================================
# 5. BÚSQUEDA + MÉTRICAS + COMPARACIÓN FINAL
# =========================================================
st.header("5. Buscar en el SRI (TF-IDF vs BM25)")

query = st.text_input("Escribe tu consulta (por ejemplo: 'covid deaths reported states'): ")

if st.button("Buscar"):
    corpus_df = st.session_state.corpus_df
    relevance = corpus_df["label_bin"].values  # 1 = real, 0 = fake

    # ==============================================
    # 5.1 Resultados TF-IDF (Top-K)
    # ==============================================
    results_tfidf, scores_top_tfidf = search_tfidf(
        query,
        st.session_state.tfidf,
        st.session_state.tfidf_matrix,
        corpus_df,
        top_k=5
    )

    st.subheader("🔹 Resultados TF-IDF (Top 5)")
    st.dataframe(results_tfidf)


    # ==============================================
    # 5.2 Resultados BM25 (Top-K)
    # ==============================================
    results_bm25, scores_top_bm25 = search_bm25(
        query,
        st.session_state.bm25,
        corpus_df,
        top_k=5
    )

    st.subheader("🔸 Resultados BM25 (Top 5)")
    st.dataframe(results_bm25)


    # ==============================================
    # 5.3 Cálculo de scores completos para métricas
    # ==============================================
    st.header("📊 Métricas del SRI (usando REAL como relevantes)")

    # Scores completos TF-IDF para TODOS los documentos
    query_vec = st.session_state.tfidf.transform([query])
    all_scores_tfidf = (st.session_state.tfidf_matrix @ query_vec.T).toarray().flatten()

    # Scores completos BM25 para TODOS los documentos
    query_tokens = query.split()
    all_scores_bm25 = st.session_state.bm25.get_scores(query_tokens)


    # -------- Métricas TF-IDF --------
    st.subheader("TF-IDF")

    st.write("Precision@5:", precision_at_k(all_scores_tfidf, relevance, k=5))
    st.write("Recall@5:",    recall_at_k(all_scores_tfidf, relevance, k=5))
    st.write("Average Precision (AP):", average_precision(all_scores_tfidf, relevance))

    cm_tfidf = sri_confusion_matrix(all_scores_tfidf, relevance)
    st.write("Matriz de Confusión (TF-IDF):")
    st.write(cm_tfidf)


    # -------- Métricas BM25 --------
    st.subheader("BM25")

    st.write("Precision@5:", precision_at_k(all_scores_bm25, relevance, k=5))
    st.write("Recall@5:",    recall_at_k(all_scores_bm25, relevance, k=5))
    st.write("Average Precision (AP):", average_precision(all_scores_bm25, relevance))

    cm_bm25 = sri_confusion_matrix(all_scores_bm25, relevance)
    st.write("Matriz de Confusión (BM25):")
    st.write(cm_bm25)


    # ==============================================
    # 5.4 COMPARACIÓN FINAL TF-IDF vs BM25
    # ==============================================
    st.header("📈 Comparación Final TF-IDF vs BM25")

    comparison_df = pd.DataFrame({
        "Métrica": ["Precision@5", "Recall@5", "Average Precision"],
        "TF-IDF": [
            precision_at_k(all_scores_tfidf, relevance, k=5),
            recall_at_k(all_scores_tfidf, relevance, k=5),
            average_precision(all_scores_tfidf, relevance)
        ],
        "BM25": [
            precision_at_k(all_scores_bm25, relevance, k=5),
            recall_at_k(all_scores_bm25, relevance, k=5),
            average_precision(all_scores_bm25, relevance)
        ]
    })

    st.dataframe(comparison_df)


    # ==============================================
    # 5.5 CONCLUSIÓN AUTOMÁTICA
    # ==============================================
    st.subheader("🧠 Conclusión Automática (SRI)")

    score_tfidf = comparison_df["TF-IDF"].sum()
    score_bm25  = comparison_df["BM25"].sum()

    if score_tfidf > score_bm25:
        st.success("➡️ **TF-IDF muestra mejor desempeño global como SRI para esta consulta (respecto a noticias reales).**")
    elif score_bm25 > score_tfidf:
        st.success("➡️ **BM25 muestra mejor desempeño global como SRI para esta consulta (respecto a noticias reales).**")
    else:
        st.info("➡️ **TF-IDF y BM25 muestran desempeño equivalente para esta consulta.**")
