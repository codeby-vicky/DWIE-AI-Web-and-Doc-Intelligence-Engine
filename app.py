import streamlit as st
from dotenv import load_dotenv
import torch
import fitz
import re
import os
import json
import hashlib
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.llms import Ollama

from core.table_extractor import extract_tables_from_pdf
from core.image_analyzer import analyze_image_with_ollama


# ==============================
# CONFIG
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_FILE = "storage/image_cache.json"

st.set_page_config(
    page_title="Hybrid Intelligence Engine",
    page_icon="📄",
    layout="wide"
)

# ==============================
# QUERY TYPE DETECTORS
# ==============================

def is_vision_query(question: str):
    vision_keywords = [
        "figure", "fig", "image", "diagram",
        "chart", "graph", "illustrate",
        "illustration", "picture"
    ]
    return any(word in question.lower() for word in vision_keywords)


def is_table_query(question: str):
    table_keywords = [
        "table", "compare", "comparison",
        "value", "accuracy", "score",
        "performance", "metric",
        "row", "column"
    ]
    return any(word in question.lower() for word in table_keywords)


def is_global_query(question: str):
    global_keywords = [
        "summarize", "summary",
        "entire paper", "full paper",
        "overall", "complete analysis",
        "whole document", "all pages"
    ]
    return any(word in question.lower() for word in global_keywords)


# ==============================
# UTILITIES
# ==============================

def hash_image(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs("storage", exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)


# ==============================
# EXTRACTION
# ==============================

def extract_page_text(page):
    text = page.get_text("text")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_web_content(url):
    documents = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator=" ")
        cleaned = re.sub(r"\s+", " ", text)

        documents.append(
            Document(
                page_content=cleaned,
                metadata={"source": url, "type": "web"}
            )
        )
    except Exception as e:
        st.error(f"URL extraction failed: {e}")

    return documents


def extract_images_from_pdf(doc, source_name, cache):
    image_docs = []
    image_data = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)

        for img in images:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image.get("image")

            if image_bytes:
                image_data.append((image_bytes, page_index + 1))

    def process_image(item):
        image_bytes, page = item
        img_hash = hash_image(image_bytes)

        if img_hash in cache:
            return cache[img_hash], page

        description = analyze_image_with_ollama(image_bytes)
        cache[img_hash] = description
        return description, page

    with ThreadPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(process_image, image_data))

    for desc, page in results:
        image_docs.append(
            Document(
                page_content=f"Image found on page {page}:\n{desc}",
                metadata={"source": source_name, "type": "image", "page": page}
            )
        )

    return image_docs


def extract_all_content(pdf_docs, url_input=None):
    all_documents = []
    cache = load_cache()

    if pdf_docs:
        for pdf in pdf_docs:
            pdf.seek(0)
            pdf_bytes = pdf.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            with ThreadPoolExecutor(max_workers=2) as executor:
                page_texts = list(executor.map(extract_page_text, doc))

            for page_number, page_text in enumerate(page_texts):
                if page_text:
                    all_documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": pdf.name,
                                "type": "text",
                                "page": page_number + 1,
                            },
                        )
                    )

            image_docs = extract_images_from_pdf(doc, pdf.name, cache)
            all_documents.extend(image_docs)

            doc.close()

            pdf.seek(0)
            table_docs = extract_tables_from_pdf(pdf)

            for t in table_docs:
                t.metadata["type"] = "table"

            all_documents.extend(table_docs)

    if url_input:
        web_docs = extract_web_content(url_input)
        all_documents.extend(web_docs)

    save_cache(cache)
    return all_documents


# ==============================
# CHUNKING
# ==============================

def get_text_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    return splitter.split_documents(documents)


# ==============================
# RETRIEVER
# ==============================

def get_hybrid_retriever(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
        encode_kwargs={"batch_size": 8}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    return EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )


# ==============================
# MODEL LOADING
# ==============================

def load_llm(mode, vision_mode=False):

    if vision_mode:
        return Ollama(model="llava:7b", temperature=0.2, num_predict=300)

    if mode == "Research Mode":
        return Ollama(model="llama3:8b-instruct-q4_0", temperature=0.2, num_predict=600)
    else:
        return Ollama(model="phi:latest", temperature=0.4, num_predict=400)


# ==============================
# ANSWER
# ==============================

def generate_answer(llm, docs, question, mode):

    MAX_DOCS = 40
    docs = docs[:MAX_DOCS]

    context = "\n\n".join(
        f"(Page {d.metadata.get('page')}) {d.page_content}"
        for d in docs
    )

    instruction = (
        "Carefully analyze all provided context. "
        "If tables are present, extract exact numeric values and compare rows and columns precisely."
        if mode == "Research Mode"
        else
        "Provide clear structured answer. Extract exact values from tables if available."
    )

    prompt = f"""
{instruction}

Context:
{context}

Question:
{question}

Answer:
"""

    return llm.invoke(prompt)


# ==============================
# CONFIDENCE
# ==============================

def confidence_score(docs):
    if not docs:
        return 0
    return min(95, 50 + len(docs) * 4)


# ==============================
# UI
# ==============================

st.title("📄 Hybrid PDF + Web Intelligence Engine")

if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = None

with st.sidebar:

    st.markdown("### 📂 Processing Panel")

    mode = st.radio("Select Mode:", ["Research Mode", "Study Mode"])
    source_mode = st.radio("Select Source:", ["Hybrid (PDF + Web)", "PDF Only", "Web Only"])

    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    url_input = st.text_input("Web URL (Optional)")

    if st.button("Process Sources"):

        if not pdf_docs and not url_input:
            st.warning("Upload PDF or provide URL.")
        else:
            with st.spinner("Processing Sources..."):
                documents = extract_all_content(pdf_docs, url_input)
                chunks = get_text_chunks(documents)
                retriever = get_hybrid_retriever(chunks)

                st.session_state.retriever = retriever
                st.session_state.all_chunks = chunks

            st.success("System Ready")


if st.session_state.retriever:

    question = st.text_input("Ask your question:")

    if question:

        vision_query = is_vision_query(question)
        table_query = is_table_query(question)
        global_query = is_global_query(question)

        with st.spinner("Generating response..."):

            if global_query:
                retrieved_docs = st.session_state.all_chunks
            else:
                retrieved_docs = st.session_state.retriever.invoke(question)

                if table_query:
                    # Force include all table chunks
                    table_docs = [
                        d for d in st.session_state.all_chunks
                        if d.metadata.get("type") == "table"
                    ]
                    retrieved_docs.extend(table_docs)

            llm = load_llm(mode, vision_mode=vision_query)

            answer = generate_answer(
                llm,
                retrieved_docs,
                question,
                mode
            )

        st.markdown("## 🧠 Answer")
        st.write(answer)

        st.markdown(f"### 📊 Confidence: {confidence_score(retrieved_docs)}%")
