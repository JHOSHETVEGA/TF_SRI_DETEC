import numpy as np

def search_tfidf(query, tfidf, tfidf_matrix, corpus, top_k=5):
    query_vec = tfidf.transform([query])
    scores = (tfidf_matrix @ query_vec.T).toarray().flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return corpus.iloc[top_idx], scores[top_idx]


def search_bm25(query, bm25, corpus, top_k=5):
    tokens = query.split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return corpus.iloc[top_idx], scores[top_idx]
