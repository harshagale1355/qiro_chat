# 🌌 Qiro Verse: Advanced AI Assistant

**Qiro Verse** is a premium, real-world AI Chatbot designed to be a high-performance, domain-aware assistant. It leverages state-of-the-art LLMs, grounded knowledge via Retrieval-Augmented Generation (RAG), and a robust context-aware state machine to execute real-world actions like restaurant bookings.

---

## ✨ Key Features

- **🧠 Contextual Memory**: Maintains full conversation history per session, allowing for natural follow-up questions and user recall.
- **📚 RAG (Retrieval-Augmented Generation)**: Grounded in domain-specific PDF data using **LangChain**, **Chromadb**, and **HuggingFace** embeddings.
- **⚡ Real-World Actions**: A sophisticated state machine that handles multi-turn interactions for booking tables, integrated with a JSON-based persistent storage backend.
- **💎 Premium UI**: A modern, high-fidelity React interface with a "Deep Space" glassmorphism aesthetic, custom animations, and responsive design.
- **🎙️ Voice-Activated**: Built-in **Speech-to-Text (STT)** and **Text-to-Speech (TTS)** support for hands-free interaction.
- **🚀 Unified Execution**: A single-script entry point to launch both the FastAPI backend and React frontend simultaneously.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI (Python 3.10)
- **AI Orchestration**: LangChain
- **LLM API**: Groq (High-speed inference)
- **Vector DB**: Chroma DB
- **Processing**: PyPDF2 / HuggingFace Transformers

### Frontend
- **Framework**: React.js + Vite
- **Styling**: Vanilla CSS (Custom Design System)
- **Capabilities**: Web Speech API (STT/TTS)

---

## 📂 Project Structure

```text
.
├── app.py              # Main FastAPI application (API Orchestrator)
├── RAG_Chatbot/        # Core Backend Logic
│   ├── backend/        # Action handlers (booking, etc.)
│   ├── components/     # RAG pipeline (retriever, embedding, LLM)
│   ├── data/           # Raw PDF knowledge sources
│   ├── logging/        # Custom logging utilities
│   └── exception/      # Custom error handling
├── frontend/           # React + Vite application
│   ├── src/
│   │   ├── App.jsx     # Main UI logic
│   │   └── index.css   # Premium glassmorphism styles
├── bookings.json       # Persistent storage for table reservations
├── run.sh              # Unified startup script
└── requirements.txt    # Python dependencies
```

---

## 🚦 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js & npm
- [Groq API Key](https://console.groq.com/)
- [OpenRouter API Key](https://openrouter.ai/) (Optional/Backup)

### 2. Installation

**Clone the repository:**
```bash
git clone <repository-url>
cd Qiro_chat
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

**Environment Variables:**
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_key_here
HF_TOKEN=your_huggingface_token_here
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

## 🔌 API Reference

### `POST /chat`
The main orchestrator. It routes incoming messages through the RAG pipeline or the booking state machine.
- **Request**: `{ "message": "string", "session_id": "string" }`
- **Response**: `{ "response": "string" }`

### `POST /action`
Triggers direct execution of backend functions.
- **Request**: `{ "action_type": "book_table", "parameters": { ... } }`

### `GET /history`
Fetches conversation history for a specific session.
- **Query Params**: `session_id`

### `GET /bookings`
Returns all successful bookings stored in the system.

---

## 📝 License
This project was developed for the **Qiro Verse Challenge**. All rights reserved.
