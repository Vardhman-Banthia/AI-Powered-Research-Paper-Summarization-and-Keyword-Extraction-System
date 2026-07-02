import collections
from nltk.tokenize import sent_tokenize
from backend.core.core.models.model_loader import load_nltk_data
from shared.constants import ACADEMIC_READING_SPEED

def compute_document_analytics(text: str) -> dict:
    if not text.strip():
        return {
            'word_count': 0, 'sentence_count': 0, 'paragraph_count': 0,
            'reading_time_minutes': 0, 'vocabulary_size': 0,
            'avg_sentence_length': 0, 'avg_word_length': 0,
            'most_frequent_terms': [], 'sentence_lengths': [], 'char_count': 0
        }
        
    load_nltk_data()
    
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    
    try:
        sentences = sent_tokenize(text)
    except:
        sentences = text.split('. ')
        
    sentence_count = len(sentences)
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    reading_time_minutes = max(1.0, round(word_count / ACADEMIC_READING_SPEED, 1))
    
    lower_words = [w.lower().strip('.,!?()[]{}"\'') for w in words]
    lower_words = [w for w in lower_words if w.isalpha()]
    
    vocabulary_size = len(set(lower_words))
    avg_sentence_length = word_count / max(1, sentence_count)
    avg_word_length = sum(len(w) for w in words) / max(1, word_count)
    
    # Filter stopwords for frequent terms
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    meaningful_words = [w for w in lower_words if w not in stop_words and len(w) > 2]
    most_frequent_terms = collections.Counter(meaningful_words).most_common(30)
    
    sentence_lengths = [len(s.split()) for s in sentences]
    
    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'paragraph_count': max(1, paragraph_count),
        'reading_time_minutes': reading_time_minutes,
        'vocabulary_size': vocabulary_size,
        'avg_sentence_length': round(avg_sentence_length, 1),
        'avg_word_length': round(avg_word_length, 1),
        'most_frequent_terms': most_frequent_terms,
        'sentence_lengths': sentence_lengths,
        'char_count': char_count
    }
