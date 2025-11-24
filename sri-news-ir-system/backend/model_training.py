import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_truth_model(train_df, val_df):
    vectorizer = CountVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val   = vectorizer.transform(val_df["text"])

    model = MultinomialNB()
    model.fit(X_train, train_df["label"])

    preds = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(val_df["label"], preds),
        "confusion": confusion_matrix(val_df["label"], preds),
        "report": classification_report(val_df["label"], preds)
    }

    return model, vectorizer, metrics
