import re
from nltk.corpus import stopwords
from backend.core.core.models.model_loader import load_spacy_model, load_nltk_data

def clean_noise(text: str) -> str:
    """Removes URLs, emails, special characters."""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove excessive newlines
    text = re.sub(r'\n+', '\n', text)
    return text

def preprocess_text(text: str) -> dict:
    load_nltk_data()
    nlp = load_spacy_model()
    
    cleaned_text = clean_noise(text.lower())
    
    doc = nlp(cleaned_text)
    
    tokens = [token.text for token in doc if not token.is_space]
    
    stop_words = set(stopwords.words('english'))
    
    lemmas = []
    for token in doc:
        if not token.is_space and not token.is_punct and not token.is_stop and token.text not in stop_words and len(token.text) >= 3:
            lemmas.append(token.lemma_)
            
    stopwords_removed = len(tokens) - len(lemmas)
    
    return {
        'cleaned_text': cleaned_text,
        'tokens': tokens,
        'lemmas': lemmas,
        'stats': {
            'token_count': len(tokens),
            'unique_tokens': len(set(tokens)),
            'stopwords_removed': stopwords_removed,
            'avg_token_length': sum(len(t) for t in tokens) / max(1, len(tokens))
        }
    }
