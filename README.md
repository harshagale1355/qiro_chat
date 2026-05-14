# 🌌 InsightMed: Advanced Medical AI Assistant

**InsightMed** is a premium, domain-aware medical AI Chatbot designed to provide high-performance, grounded health information. It leverages state-of-the-art LLMs and a robust Retrieval-Augmented Generation (RAG) pipeline to accurately answer complex medical queries, such as questions about diabetes, hypertension, and other health conditions.

Recently refactored, the project now features an optimized pipeline using **Qdrant** for vector storage, **LCEL** (LangChain Expression Language) for dynamic orchestration, a fine-tuned medical embedding model, and **LangSmith** for deep observability.

---

## ✨ Key Features & Architecture Updates

- **🧠 Contextual Memory**: Maintains full conversation history per session, allowing for natural follow-up questions and symptom tracking across the conversation.
- **📚 Advanced RAG with Hybrid Search**: Grounded in domain-specific medical PDF data. Migrated from ChromaDB to **Qdrant** for better scalability, utilizing both dense and sparse vectors for robust **Hybrid Search (RRF Fusion)**.
- **⚙️ LCEL Orchestration**: The entire LangChain pipeline has been rewritten using LangChain Expression Language (LCEL) for a modular, readable, and highly maintainable workflow.
- **🔬 LangSmith Tracing**: Full integration with **LangSmith** provides granular observability, performance monitoring, and debugging for every step of the retrieval and generation process.
- **🧬 Fine-Tuned Embedding Model**: Replaced standard embeddings with a custom **fine-tuned SentenceTransformer model** to achieve higher semantic accuracy for complex medical queries.
- **💎 Premium UI**: A modern, high-fidelity React interface with a "Deep Space" glassmorphism aesthetic, custom animations, responsive design, and robust API integration handling loading and error states seamlessly.
- **🎙️ Voice-Activated**: Built-in **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** support for hands-free interaction, ensuring accessibility.
- **🚀 Unified Execution**: A single-script entry point to launch both the FastAPI backend and React frontend simultaneously.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.10)
- **AI Orchestration**: LangChain (LCEL implementation)
- **LLM API**: Groq (High-speed inference via `ChatGroq`)
- **Vector Database**: Qdrant (Hybrid Search capable)
- **Embeddings**: Fine-Tuned SentenceTransformers & FastEmbed (Sparse)
- **Observability**: LangSmith
- **Processing**: PyPDF2 / HuggingFace Transformers

### Frontend
- **Framework**: React.js + Vite
- **Styling**: Vanilla CSS (Custom Design System with Glassmorphism)
- **Capabilities**: Web Speech API (STT/TTS)
- **API Integration**: Dynamic REST connection via environment variables

---

## 📂 Project Structure

```text
.
├── app.py              # Main FastAPI application (API Orchestrator)
├── RAG_Chatbot/        # Core Backend Logic
│   ├── components/     # RAG pipeline elements
│   │   ├── LLM/        # LLMManager (ChatGroq initialization)
│   │   ├── retriever/  # QdrantRetriever & LCEL pipeline configuration
│   │   └── fine_tune_embed/ # Fine-tuned medical embedding models
│   ├── data/           # Raw medical knowledge sources
│   ├── logging/        # Custom logging utilities
│   └── exception/      # Custom error handling
├── frontend/           # React + Vite application
│   ├── src/
│   │   ├── App.jsx     # Main UI logic (End-to-end API integrations)
│   │   └── index.css   # Premium glassmorphism styles
├── qdrant_storage/     # Local persistence for Qdrant Vector DB
├── run.sh              # Unified startup script
└── requirements.txt    # Python dependencies
```

---

## 🔌 Qdrant Setup & Fine-Tuned Model Details

### Qdrant Vector Database
The project utilizes **Qdrant** for its hybrid search capabilities. The local database state is persisted in the `./qdrant_storage` directory. 
- **Dense Vectors**: Handled by the fine-tuned medical model.
- **Sparse Vectors**: Powered by `Qdrant/bm25` (via `fastembed`).
- **Fusion**: Uses Reciprocal Rank Fusion (RRF) for the best balance between keyword matches and semantic similarity.

### Fine-Tuned Embeddings
We utilize a domain-optimized `SentenceTransformer` loaded from `RAG_Chatbot/components/fine_tune_embed/fine_tuned_model`. This ensures medical queries strictly map to our custom dataset better than off-the-shelf embedding models.

---

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js & npm
- [Groq API Key](https://console.groq.com/)
- [LangSmith API Key](https://smith.langchain.com/)

### 2. Installation

**Clone the repository:**
```bash
git clone <repository-url>
cd InsightMed
```

**Set up Backend (Virtual Environment):**
```bash
# Create and activate venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Set up Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 3. Environment Variables

Create a `.env` file in the root directory and configure the following:

```env
# Core AI Services
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_huggingface_token_here

# LangSmith Observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT="InsightMed_Chatbot"
```

To configure the frontend to talk to a remote or specific backend URL (optional), create `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
```

---

## 🚀 Running the App

We have simplified the process into a single command:

1. **Make the script executable:**
   ```bash
   chmod +x run.sh
   ```

2. **Launch the platform:**
   ```bash
   ./run.sh
   ```

The application will be available at:
- **UI**: [http://localhost:3000](http://localhost:3000)
- **Backend API**: [http://localhost:8000](http://localhost:8000)
- **API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🌐 Frontend & Backend Integration
The React frontend has been rigorously structured to integrate cleanly with the FastAPI backend:
- **Resilient API Calls**: `App.jsx` handles seamless POST and GET operations to the REST endpoints.
- **Loading & State Management**: Gracefully handles "typing" indicators (`isTyping`), API errors, and conversation history rehydration on mount.
- **Environment Agnostic**: The frontend uses `import.meta.env.VITE_API_URL` to dynamically map backend URLs for production readiness.
- **Medical Answers Rendered**: Server responses map cleanly to chat bubbles, giving the user an intuitive conversational healthcare experience.

---

## 🔌 API Reference

### `POST /chat`
The main LCEL orchestrator. It routes incoming messages through the Qdrant RAG pipeline to provide robust medical answers.
- **Request**: `{ "message": "string", "session_id": "string" }`
- **Response**: `{ "response": "string" }`

### `GET /history`
Fetches conversation history for a specific session.
- **Query Params**: `?session_id=default_user`

---

## 🔮 Future Improvements

- **Authentication**: Implementing OAuth 2.0 / JWT for multi-user, secured session tracking.
- **Medical Tools Integration**: Expanding the agent to integrate with external health APIs, symptom checkers, and appointment scheduling systems.
- **Frontend Refinements**: Deploying the Vite frontend to a CDN (Vercel/Netlify) and dockerizing the backend for completely isolated production setups.
- **Advanced Agentic Routing**: Adding LangGraph for dynamic self-reflection, multi-agent workflows, and specialized sub-agents (e.g., triage vs generic inquiries).

---

## 📝 License
This project was developed for the **InsightMed AI Application**. All rights reserved.
