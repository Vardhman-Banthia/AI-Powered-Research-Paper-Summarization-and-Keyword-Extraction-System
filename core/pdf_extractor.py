import io
import pdfplumber
import PyPDF2

def extract_text_pdfplumber(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
    return text

def extract_text_pypdf2(file_bytes: bytes) -> str:
    text = ""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

def extract_text(file_bytes: bytes) -> dict:
    text = ""
    page_count = 0
    metadata = {}
    pages = []
    
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            metadata = pdf.metadata
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
                    text += page_text + "\n\n"
    except Exception as e:
        # Fallback to PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            page_count = len(pdf_reader.pages)
            metadata = pdf_reader.metadata if pdf_reader.metadata else {}
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
                    text += page_text + "\n\n"
        except Exception as fallback_e:
            raise ValueError(f"Could not extract text. Plumber error: {e}. PyPDF2 error: {fallback_e}")
            
    return {
        'text': text.strip(),
        'page_count': page_count,
        'metadata': metadata,
        'pages': pages
    }
