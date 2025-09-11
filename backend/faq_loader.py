import csv
import os
from confidence import compute_confidence

FAQ_PATH = os.path.join(os.path.dirname(__file__), "data", "faqs.csv")

# Load FAQs once at startup
faq_list = []
with open(FAQ_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        faq_list.append({
            "question": row["question"],
            "answer": row["answer"]
        })

def load_faq():
    """Return all FAQs"""
    return faq_list

def find_best_faq(user_question: str):
    """Find the most similar FAQ entry based on confidence score."""
    best_match = None
    best_conf = 0.0

    for faq in faq_list:
        score = compute_confidence(user_question, faq["question"])
        if score > best_conf:
            best_conf = score
            best_match = faq

    if best_match:
        return best_match["answer"], best_conf
    return None, 0.0
