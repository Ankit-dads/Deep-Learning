import pickle
import os
import re
import numpy as np

# Paths to model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'spam_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')

# Load models once at module import (singleton pattern)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, 'rb') as f:
    vectorizer = pickle.load(f)


def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)      # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)              # remove emails
    text = re.sub(r'[^a-z\s]', ' ', text)             # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()          # collapse whitespace
    return text


def get_top_keywords(text_vec, n=8):
    """Return top N TF-IDF weighted feature names for the input vector."""
    feature_names = vectorizer.get_feature_names_out()
    scores = text_vec.toarray()[0]
    top_indices = scores.argsort()[::-1][:n]
    return [feature_names[i] for i in top_indices if scores[i] > 0]


def predict(text: str) -> dict:
    """
    Run spam prediction on a single text.
    Returns a dict with:
      - label: 'spam' | 'ham'
      - confidence: float (0-100)
      - top_keywords: list[str]
      - cleaned_text: str
    """
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]   # [ham_prob, spam_prob]
    pred_label_idx = int(np.argmax(proba))
    label = 'spam' if pred_label_idx == 1 else 'ham'
    confidence = round(float(proba[pred_label_idx]) * 100, 2)
    top_keywords = get_top_keywords(vec)
    return {
        'label': label,
        'confidence': confidence,
        'top_keywords': top_keywords,
        'cleaned_text': cleaned,
        'spam_prob': round(float(proba[1]) * 100, 2),
        'ham_prob': round(float(proba[0]) * 100, 2),
    }


def predict_bulk(texts: list) -> list:
    """Run predictions on a list of texts, return list of result dicts."""
    return [predict(t) for t in texts if t.strip()]
