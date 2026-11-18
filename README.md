# 🛡️ Enterprise Multi-Agent RAG Knowledge Vault

An enterprise-grade, agentic Retrieval-Augmented Generation (RAG) and validation system built with Python, Streamlit, LangChain, and ChromaDB. Designed to provide high-accuracy internal knowledge retrieval while eliminating LLM hallucinations through automated agentic routing and guardrail validation.

🌐 **Live Demo:** [View Live Streamlit Cloud Application](https://enterprise-rag-vault-6bj3y3aeyququqb6sc5mfm.streamlit.app/)

---

## 🎯 Project Overview

In enterprise environments, standard Retrieval-Augmented Generation (RAG) pipelines often fail due to blind text retrieval, lack of query intent classification, and vulnerability to hallucinations. The **Enterprise Multi-Agent RAG Knowledge Vault** solves these challenges by combining a modular multi-agent workflow with a persistent vector database and a responsive dark-mode Tailwind UI.

---

## 🏗️ Core Architecture & Agentic Workflow

The system operates across three specialized agentic layers:

1. **Agentic Router (`AgentRouter`)**
   * Dynamically inspects and classifies incoming user queries.
   * Routes requests to specialized execution paths (e.g., HR Policy Vault vs. Financial DB) to ensure efficient, domain-specific retrieval.

2. **Retrieval Pipeline (`similarity_search`)**
   * Utilizes LangChain's `RecursiveCharacterTextSplitter` (`chunk_size=500`, `chunk_overlap=50`) to maintain semantic continuity across document boundaries.
   * Leverages HuggingFace embeddings (`all-MiniLM-L6-v2`) for high-performance, lightweight vector encoding backed by **ChromaDB**.

3. **Validator Agent (`ValidatorAgent`)**
   * Acts as a hallucination guardrail layer.
   * Evaluates similarity scores and cross-references retrieved chunks against the user query before generating verified responses, flagging low-confidence matches.

---

## 📂 Project Structure

```text
enterprise-rag-vault/
│
├── app.py               # Main Streamlit application & integrated custom UI
├── requirements.txt     # Python package dependencies
├── .gitignore           # Git ignore file for virtual envs and caches
├── LICENSE              # MIT Open-Source License
└── chroma_db/           # Persistent local vector database storage
