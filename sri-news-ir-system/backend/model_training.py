import pandas as pd
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ----------------------------------------------
# LIMPIEZA DE TEXTO
# ----------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+", "", text)        
    text = re.sub(r"@\w+", "", text)          
    text = re.sub(r"[^a-zA-ZñÑáéíóú ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ----------------------------------------------
# LIMPIEZA COMPLETA DEL DATAFRAME
# ----------------------------------------------
def clean_dataframe(df):
    df["tweet"] = df["tweet"].astype(str).apply(clean_text)

    # Mapeo oficial de tu dataset
    mapping = {"fake": 0, "real": 1}
    df["label"] = df["label"].map(mapping)

    # Eliminar NA
    df = df.dropna(subset=["tweet", "label"])
    df["label"] = df["label"].astype(int)

    return df


# ----------------------------------------------
# ENTRENAR MODELO REAL vs FAKE
# ----------------------------------------------
def train_truth_model(train_df, val_df):

    train_df = clean_dataframe(train_df)
    val_df   = clean_dataframe(val_df)

    vectorizer = CountVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(train_df["tweet"])
    X_val   = vectorizer.transform(val_df["tweet"])

    model = MultinomialNB()
    model.fit(X_train, train_df["label"])

    preds = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(val_df["label"], preds),
        "confusion": confusion_matrix(val_df["label"], preds),
        "report": classification_report(val_df["label"], preds)
    }

    return model, vectorizer, metrics
