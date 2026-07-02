from backend.core.core.models.model_loader import load_keybert_model
from shared.constants import KEYWORD_NGRAM_RANGE, KEYWORD_DIVERSITY
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords_keybert(text: str, top_n: int = 10) -> list[dict]:
    if not text.strip():
        return []
    model = load_keybert_model()
    keywords = model.extract_keywords(
        text, 
        keyphrase_ngram_range=KEYWORD_NGRAM_RANGE, 
        stop_words='english', 
        use_mmr=True, 
        diversity=KEYWORD_DIVERSITY, 
        top_n=top_n
    )
    return [{'keyword': kw[0], 'score': float(kw[1]), 'method': 'KeyBERT'} for kw in keywords]

def extract_keywords_tfidf(text: str, top_n: int = 10) -> list[dict]:
    if not text.strip():
        return []
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=KEYWORD_NGRAM_RANGE)
    try:
        tfidf_matrix = vectorizer.fit_transform([text])
    except ValueError:
        return [] # empty vocabulary
        
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.toarray()[0]
    
    # Get frequencies
    words = text.lower().split()
    from collections import Counter
    word_counts = Counter(words)
    
    top_indices = np.argsort(scores)[::-1][:top_n]
    
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            kw = feature_names[idx]
            # Approximate frequency for n-grams by checking occurrences in string
            freq = text.lower().count(kw)
            results.append({'keyword': kw, 'score': float(scores[idx]), 'frequency': freq, 'method': 'TF-IDF'})
            
    return results

def extract_all_keywords(text: str, top_n: int = 10) -> dict:
    return {
        'keybert': extract_keywords_keybert(text, top_n),
        'tfidf': extract_keywords_tfidf(text, top_n)
    }
