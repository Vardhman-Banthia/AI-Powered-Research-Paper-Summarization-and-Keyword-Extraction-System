import subprocess
import sys
import nltk
from keybert import KeyBERT
import spacy
from functools import lru_cache
from shared.constants import BART_MODEL_NAME, T5_MODEL_NAME, KEYBERT_MODEL_NAME, SPACY_MODEL_NAME
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@lru_cache(maxsize=2)
def load_summarization_model(model_name: str):
    print(f"Loading summarization model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return {"model": model, "tokenizer": tokenizer}

@lru_cache(maxsize=1)
def load_keybert_model():
    print("Loading KeyBERT model...")
    return KeyBERT(model=KEYBERT_MODEL_NAME)

@lru_cache(maxsize=1)
def load_spacy_model():
    print("Loading SpaCy model...")
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL_NAME])
            nlp = spacy.load(SPACY_MODEL_NAME)
        return nlp

@lru_cache(maxsize=1)
def load_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    return True

def get_model_info() -> dict:
    return {
        "BART": BART_MODEL_NAME,
        "T5": T5_MODEL_NAME,
        "KeyBERT": KEYBERT_MODEL_NAME,
        "SpaCy": SPACY_MODEL_NAME
    }
