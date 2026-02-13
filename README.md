# DWIE AI – Web & Document Intelligence Engine

DWIE AI (Document & Web Intelligence Engine) is a Hybrid AI-powered Retrieval and Reasoning System designed to perform intelligent document analysis, semantic search, and context-grounded response generation across PDF-based knowledge sources.

This document is written as a **complete technical revision guide**. If you revisit this project after one year, this README alone should help you understand the architecture, logic, design decisions, and execution flow without re-learning from scratch.

---

# 1. Project Vision

The objective of DWIE AI is to build a grounded document intelligence engine that:

• Extracts structured and unstructured data from PDFs
• Converts text into semantic vector embeddings
• Stores embeddings efficiently
• Retrieves context using similarity search
• Generates accurate responses strictly based on indexed content

The system eliminates hallucination by grounding all responses in retrieved document chunks.

---

# 2. Core Architecture Overview

The system follows a Hybrid Retrieval + LLM Grounding pipeline.

High-Level Flow:

User Query → Embedding → Vector Search → Context Selection → Prompt Injection → LLM Response

---

# 3. End-to-End Execution Flow

## Step 1: Document Ingestion

* User uploads PDF(s)
* PDF parser extracts:

  * Raw text
  * Tables
  * Images (if applicable)

Relevant modules:

* core/extractor.py
* core/table_extractor.py
* core/image_analyzer.py

Purpose:
Convert document into machine-processable content.

---

## Step 2: Text Chunking

Large text is divided into manageable chunks.

Why?
LLMs and vector stores perform better with optimized chunk sizes.

Typical chunk strategy:

* Fixed token length
* Overlapping windows to preserve context

Chunking ensures:

* Better retrieval accuracy
* Reduced memory overhead
* Faster similarity computation

---

## Step 3: Embedding Generation

Each text chunk is converted into a dense vector representation.

Embedding transforms text into numerical form that captures semantic meaning.

Output:
Vector array representing document knowledge space.

---

## Step 4: Vector Storage (FAISS)

Vectors are stored inside a FAISS index.

FAISS enables:

* Fast nearest-neighbor search
* Efficient similarity comparison
* Scalable vector retrieval

Storage location:

* storage/faiss_index/

Note:
Index files are generated dynamically and are excluded from Git tracking.

---

## Step 5: Query Processing

When user submits a question:

1. Question is embedded
2. FAISS retrieves top-k similar chunks
3. Most relevant contexts are selected

This ensures semantic alignment between question and document knowledge.

---

## Step 6: Context Grounding

Retrieved chunks are injected into a structured prompt template.

Prompt structure:

Context: <retrieved chunks>

Question: <user query>

Instruction:
Answer strictly using the provided context.

This prevents hallucinated answers.

---

## Step 7: Response Generation

The Language Model processes the grounded prompt and generates:

* Context-aware answer
* Accurate explanation
* Structured output if required

The response is displayed through Streamlit web interface.

---

# 4. Technical Stack

Programming Language:

* Python

Core Libraries:

* FAISS (Vector Search)
* Streamlit (Web Interface)
* PDF Processing Libraries
* Embedding Model
* LLM Backend (OpenAI or Local Model)

Architecture Type:
Hybrid Retrieval-Augmented Generation (RAG)

---

# 5. Project Directory Breakdown

DWIE-AI-Web-and-Doc-Intelligence-Engine/

core/
extractor.py          → PDF text extraction
table_extractor.py    → Table parsing
image_analyzer.py     → Image-level analysis
retriever.py          → FAISS retrieval logic
prompts.py            → Prompt templates
models.py             → Model initialization
evaluation.py         → Performance validation
performance.py        → Benchmark utilities

storage/
faiss_index/          → Generated vector index (ignored in Git)

app.py                  → Application entry point
htmlTemplates.py        → UI layout logic
requirements.txt        → Dependency list
.gitignore              → Artifact exclusion
README.md               → Technical documentation

---

# 6. Installation Guide

## 1. Clone Repository

```
git clone https://github.com/codeby-vicky/DWIE-AI-Web-and-Doc-Intelligence-Engine.git
cd DWIE-AI-Web-and-Doc-Intelligence-Engine
```

## 2. Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```
pip install -r requirements.txt
```

---

# 7. Environment Configuration

If using external API:

Create .env file:

OPENAI_API_KEY=your_api_key

If using local LLM:

Ensure local model server is running before launching app.

---

# 8. Running the Application

```
streamlit run app.py
```

System launches in browser.

User Workflow:

1. Upload PDFs
2. Wait for indexing
3. Ask natural language questions
4. Receive grounded responses

---

# 9. Design Principles

• Grounded Intelligence (No blind generation)
• Retrieval before reasoning
• Separation of concerns (Extraction / Retrieval / Generation)
• Modular architecture
• Storage isolation

---

# 10. Performance Considerations

Key performance factors:

* Chunk size
* Embedding model quality
* Top-k retrieval size
* Index type
* Prompt engineering

Optimization strategies:

* Reduce redundant chunks
* Use overlap carefully
* Optimize FAISS index type
* Cache embeddings

---

# 11. Known Limitations

* Dependent on embedding quality
* Requires re-indexing for new documents
* Performance tied to hardware
* Complex tables may require enhanced parsing

---

# 12. Future Roadmap

• Cross-document reasoning
• Knowledge graph integration
• API deployment layer
• Docker containerization
• Cloud deployment
• Multi-modal reasoning upgrades

---

# 13. If Revising After One Year

To quickly recall system logic:

1. Understand RAG architecture
2. Review extractor.py → ingestion
3. Review retriever.py → FAISS logic
4. Review prompts.py → grounding strategy
5. Review app.py → execution flow

This restores full system understanding within minutes.

---

# 14. License

Released under MIT License.

---

DWIE AI represents a modular, scalable foundation for enterprise-grade document intelligence systems.
