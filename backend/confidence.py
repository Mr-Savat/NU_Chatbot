# backend/confidence.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reusable vectorizer
vectorizer = TfidfVectorizer()

def compute_confidence(user_question: str, faq_question: str) -> float:
    """
    Compute confidence score (0 to 1) using cosine similarity
    between user input and an FAQ question.
    """
    texts = [user_question, faq_question]
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return float(similarity[0][0])
