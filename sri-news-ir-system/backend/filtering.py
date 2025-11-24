def filter_real_news(model, vectorizer, test_df):
    X = vectorizer.transform(test_df["tweet"])
    test_df["prediction"] = model.predict(X)

    real_df = test_df[test_df["prediction"] == 1].reset_index(drop=True)
    return real_df
