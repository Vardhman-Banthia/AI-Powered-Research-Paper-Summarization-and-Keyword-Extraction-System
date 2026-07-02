from backend.core.core.models.model_loader import load_summarization_model
from utils.text_utils import chunk_text
from shared.constants import BART_MODEL_NAME, T5_MODEL_NAME

class Summarizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        components = load_summarization_model(model_name)
        self.model = components["model"]
        self.tokenizer = components["tokenizer"]
        
    def summarize(self, text: str, max_length: int = 200, min_length: int = 80) -> str:
        if not text.strip():
            return ""
        if len(text.split()) < min_length:
            return text
            
        chunks = chunk_text(text, max_tokens=1024)
        if len(chunks) == 1:
            return self._summarize_chunk(chunks[0], max_length, min_length)
        else:
            return self._summarize_long_text(chunks, max_length, min_length)
            
    def _summarize_chunk(self, text: str, max_length: int, min_length: int) -> str:
        # Approximate input token length
        input_len = len(text.split())
        adj_max = min(max_length, max(min_length + 10, int(input_len * 0.8)))
        adj_min = min(min_length, int(input_len * 0.2))
        
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=adj_max, 
            min_length=adj_min, 
            do_sample=False, 
            num_beams=4, 
            no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
    def _summarize_long_text(self, chunks: list[str], max_length: int, min_length: int) -> str:
        chunk_summaries = []
        # Calculate per-chunk target lengths to sum up to max_length roughly
        per_chunk_max = max(50, int(max_length / len(chunks)))
        per_chunk_min = max(20, int(min_length / len(chunks)))
        
        for chunk in chunks:
            if len(chunk.split()) > 20:
                summary = self._summarize_chunk(chunk, per_chunk_max, per_chunk_min)
                chunk_summaries.append(summary)
                
        combined_summary = " ".join(chunk_summaries)
        
        # If combined is still too long, summarize it again
        if len(combined_summary.split()) > max_length * 1.5:
            combined_summary = self._summarize_chunk(combined_summary, max_length, min_length)
            
        return combined_summary

_summarizers = {}

def get_bart_summarizer() -> Summarizer:
    if "BART" not in _summarizers:
        _summarizers["BART"] = Summarizer(BART_MODEL_NAME)
    return _summarizers["BART"]

def get_t5_summarizer() -> Summarizer:
    if "T5" not in _summarizers:
        _summarizers["T5"] = Summarizer(T5_MODEL_NAME)
    return _summarizers["T5"]
