from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf(kmer_lists, k):
    tfidf_vectorizer = TfidfVectorizer(analyzer='char', lowercase=False, ngram_range=(k, k))
    tfidf_matrix = tfidf_vectorizer.fit_transform(kmer_lists)
    return tfidf_matrix