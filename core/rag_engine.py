import os
import sys
import traceback
from typing import List, Dict, Tuple, Callable
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from backend.core.config import GEMINI_API_KEY, validate_config

class RAGInitializationError(Exception):
    """Custom exception raised when the RAG pipeline fails to initialize."""
    def __init__(self, step_name: str, original_exception: Exception):
        self.step_name = step_name
        self.original_exception = original_exception
        self.reason = f"{type(original_exception).__name__}: {str(original_exception)}"
        super().__init__(f"Failed During: {step_name} -> {self.reason}")

def _log_step_success(step_num: int, total_steps: int, msg: str):
    print(f"[{step_num}/{total_steps}] {msg}...\nSUCCESS\n", file=sys.stderr)

def _log_step_failure(step_num: int, total_steps: int, msg: str, e: Exception, variables: dict = None):
    print(f"[{step_num}/{total_steps}] {msg}...\nFAILED\n", file=sys.stderr)
    
    exc_type = type(e).__name__
    exc_msg = str(e)
    
    # Safely extract traceback line info
    tb = getattr(e, "__traceback__", None)
    if tb:
        tb_info = traceback.extract_tb(tb)[-1]
        filename = tb_info.filename
        lineno = tb_info.lineno
        func_name = tb_info.name
    else:
        filename = "Unknown"
        lineno = "Unknown"
        func_name = "Unknown"
        
    print("================ ERROR DIAGNOSTIC ================", file=sys.stderr)
    print(f"Step:       {msg}", file=sys.stderr)
    print(f"Exception:  {exc_type}: {exc_msg}", file=sys.stderr)
    print(f"File:       {filename}", file=sys.stderr)
    print(f"Line:       {lineno}", file=sys.stderr)
    print(f"Function:   {func_name}", file=sys.stderr)
    if variables:
        print("Variables:", file=sys.stderr)
        for k, v in variables.items():
            val_rep = f"len={len(v)}" if hasattr(v, '__len__') and not isinstance(v, str) else str(v)
            print(f"  {k} = {type(v).__name__} ({val_rep})", file=sys.stderr)
    print("==================================================", file=sys.stderr)

def setup_rag_pipeline(pages_list: List[str], progress_callback: Callable[[str], None] = None) -> Tuple[Chroma, int]:
    """
    Sets up the RAG pipeline by chunking text, creating embeddings,
    and storing them in an in-memory Chroma vector database.
    """
    total = 5
    def report(msg):
        if progress_callback:
            progress_callback(msg)

    # 1. Validation
    try:
        report("Loading Configuration")
        if not validate_config():
            raise ValueError("API key is invalid or missing in .env")
        _log_step_success(1, total, "Loading environment variables & API Key")
    except Exception as e:
        _log_step_failure(1, total, "Loading environment variables", e)
        raise RAGInitializationError("Loading Configuration", e)

    # 2. Reading PDF
    try:
        report("Reading PDF")
        documents = []
        for i, page_text in enumerate(pages_list):
            if page_text.strip():
                documents.append(Document(page_content=page_text, metadata={"page": i + 1}))
        if not documents:
            raise ValueError("No text found in the document to process.")
        _log_step_success(2, total, "Reading uploaded PDF")
    except Exception as e:
        _log_step_failure(2, total, "Reading uploaded PDF", e, {"pages_list_len": len(pages_list)})
        raise RAGInitializationError("Reading PDF", e)

    # 3. Splitting Text
    try:
        report("Splitting Text")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Log Statistics
        total_chars = sum(len(d.page_content) for d in documents)
        total_chunks = len(chunks)
        avg_chunk_size = sum(len(c.page_content) for c in chunks) // total_chunks if total_chunks > 0 else 0
        
        print("\n--- CHUNKING STATISTICS ---", file=sys.stderr)
        print(f"Total Pages: {len(documents)}", file=sys.stderr)
        print(f"Total Characters: {total_chars:,}", file=sys.stderr)
        print(f"Total Chunks: {total_chunks:,}", file=sys.stderr)
        print(f"Average Chunk Size: {avg_chunk_size:,}", file=sys.stderr)
        print("---------------------------\n", file=sys.stderr)
        
        _log_step_success(3, total, "Splitting text")
    except Exception as e:
        _log_step_failure(3, total, "Splitting text", e, {"documents_len": len(documents)})
        raise RAGInitializationError("Splitting Text", e)
        
    # 4. Creating Embeddings
    try:
        report("Creating Embeddings")
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2", 
            google_api_key=GEMINI_API_KEY
        )
        
        # Test auth to catch failures early
        embeddings.embed_query("authentication check")
        
        # Log each chunk being prepared
        print("\n--- EMBEDDING LOGS ---", file=sys.stderr)
        for i, chunk in enumerate(chunks):
            print(f"Embedding Chunk {i + 1}...", file=sys.stderr)
        print("----------------------\n", file=sys.stderr)
        
        _log_step_success(4, total, "Creating embeddings")
    except Exception as e:
        _log_step_failure(4, total, "Creating embeddings", e, {"model": "models/gemini-embedding-2"})
        raise RAGInitializationError("Creating Embeddings", e)

    # 5. Building DB
    try:
        report("Building Knowledge Base")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        _log_step_success(5, total, "Creating Chroma Vector Database")
    except Exception as e:
        _log_step_failure(5, total, "Creating Chroma Vector Database", e, {"chunks_len": len(chunks)})
        raise RAGInitializationError("Building Knowledge Base", e)

    return vectorstore, len(chunks)

def generate_answer(vectorstore, question: str, chat_history: List[Dict], mode: str) -> Tuple[str, List[Document]]:
    """
    Retrieves context and generates an answer using Gemini 2.5 Flash.
    Adjusts tone based on the mode.
    """
    if not validate_config():
        raise ValueError("Configuration Error: API key is invalid or missing.")
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=GEMINI_API_KEY, 
        temperature=0.2
    )
    
    # Retrieve relevant documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    
    # Format context
    context = ""
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "Unknown")
        context += f"--- Chunk {i+1} (Page {page}) ---\n{doc.page_content}\n\n"
        
    # Build System Prompt based on Mode
    if mode == "Beginner":
        system_instruction = (
            "You are a helpful, enthusiastic AI Research Assistant. "
            "Explain the concepts as if you are talking to a beginner or a student. "
            "Use simple language, avoid overly dense academic jargon where possible, "
            "and use analogies if helpful. "
            "Answer the question based ONLY on the provided context from the research paper. "
            "If the context does not contain the answer, politely state that it's not discussed in the paper."
        )
    else:
        system_instruction = (
            "You are an expert AI Research Scientist and Academic Reviewer. "
            "Provide highly technical, precise, and rigorously formal answers. "
            "Use appropriate academic terminology. "
            "Answer the question based ONLY on the provided context from the research paper. "
            "If the context does not contain the answer, explicitly state that the provided text lacks this information."
        )
        
    # Construct the message history for LangChain
    messages = [SystemMessage(content=f"{system_instruction}\n\nContext:\n{context}")]
    
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
            
    messages.append(HumanMessage(content=question))
    
    # Generate the response
    response = llm.invoke(messages)
    
    return response.content, docs
