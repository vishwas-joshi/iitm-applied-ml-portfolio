# RAG Pipeline: Evalulation of the Robustness of Machine Learning Surrogate Models for Wind Turbine Fatigue Load Estimation using SCADA data

A modular, privacy-focused Retrieval-Augmented Generation (RAG) pipeline designed to extract engineering insights from complex technical documents. 

This repository demonstrates the pipeline in action using a published **TORQUE conference research paper on validating machine learning load emulator architectures** across wind turbine configurations. By utilizing this domain-specific application, this project showcases both modern RAG architecture and applied expertise in Ultimate Limit State (ULS) and Fatigue Limit State (FLS) structural load analysis.

---

## 🏗️ Architecture & Tech Stack

- **Orchestration:** LangChain (LCEL)
- **Vector Database:** ChromaDB (Local, persistent storage)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local inference)
- **LLM Engine:** Groq API (`llama-3.3-70b-versatile`)
- **Document Processing:** PyPDFLoader, RecursiveCharacterTextSplitter

---

## ✨ Key Features & Extensibility

* **Strict Data Privacy:** Documents are embedded and stored locally on disk. The LLM only receives isolated, highly relevant text chunks rather than the entire proprietary document, avoiding closed-ecosystem external services.
* **Zero-Hallucination Guardrails:** Implements a strict Prompt Template enforcing deterministic outputs (Temperature = 0) and instructing the model to admit ignorance if the retrieved context is insufficient.
* **Maximum Marginal Relevance (MMR):** The vector retriever is configured to balance semantic similarity with document diversity, preventing redundant context extraction.
* **Fully Extensible:** While the primary showcase queries structural dynamics and load emulator methodology, the pipeline is entirely document-agnostic. The `DocumentProcessor` can ingest any technical PDF by simply updating the target file path.
* **Enterprise Environment Compatibility:** Engineered with a custom pure-Python dependency mock (`uuid_utils`) to bypass strict Windows Application Control policies that block compiled C-extensions.

---

## 📂 Project Structure

```text
rag/
├── .env                       # Environment variables (GROQ_API_KEY)
├── .gitignore                 # Excludes venv, databases, and secrets
├── README.md                  # Project documentation
├── requirements.txt           # Curated dependency list
├── data/
│   ├── chroma_db/             # Local vector database storage
│   └── torque_research_paper.pdf  # Source document for ingestion
├── notebooks/
│   └── 01_rag_pipeline.ipynb  # Executive orchestration layer
├── src/
│   ├── data.py                # Document loading and chunking 
│   ├── test_data.py           # Unit tests
│   ├── vector_store.py        # Embeddings and MMR retriever 
│   ├── test_vector_store.py   # Unit tests
│   ├── model.py               # LLM initialization and LCEL chain 
│   ├── test_model.py          # Unit tests
│   └── uuid_utils/            # Pure-Python dependency mock
│       ├── __init__.py
│       └── compat.py
└── venv/                      # Isolated virtual environment


## 🚀 Setup & Installation

**1. Clone the repository and navigate to the project directory:**
```bash
git clone <your-repo-url>
cd rag
```

**2. Create and activate a Python virtual environment:**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Configure Environment Variables:**
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=gsk_your_api_key_here
```

**5. Place Source Data:**
Add your target PDF (e.g., the TORQUE methodology paper) to the `data/` directory.