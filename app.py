import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from RAG_Chatbot.components.retriever.retriever import QdrantRetriever
from RAG_Chatbot.components.LLM.LLM import LLMManager
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

app = FastAPI(title="InsightMed Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    retriever = QdrantRetriever(limit=3)
    llm_general = LLMManager().get_model()
    
    template = """Answer the question based only on the following medical context:
{context}

Question: {question}
"""
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_general
        | StrOutputParser()
    )

except Exception as e:
    print("WARNING: Could not initialize QAchain/LLM on startup.", e)
    qa_chain = None
    llm_general = None

# Stores the full chat history per session
history_store: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


def is_small_talk(message: str) -> bool:
    greetings = ["hi", "hello", "hey", "good morning", "how are you"]
    return message.lower().strip() in greetings


def build_llm_messages(session_id: str, current_user_msg: str) -> list:
    """
    Constructs the message list for the LLM, including:
      - A system prompt
      - The full conversation history for this session
      - The current user message
    """
    system_prompt = (
        "You are the InsightMed AI Assistant, a medical chatbot. "
        "Be helpful, friendly, and concise. "
        "If the user asks questions about their health, diseases like diabetes or hypertension, "
        "answer using your medical knowledge and the conversation history provided."
    )

    messages = [("system", system_prompt)]

    # Replay the full conversation history so the LLM has context
    for turn in history_store.get(session_id, []):
        role = "human" if turn["role"] == "user" else "assistant"
        messages.append((role, turn["content"]))

    return messages


@app.post("/chat")
async def chat(req: ChatRequest):
    user_msg = req.message.strip()
    session_id = req.session_id

    if session_id not in history_store:
        history_store[session_id] = []

    # Persist the user's message first (so history is complete for LLM calls)
    history_store[session_id].append({"role": "user", "content": user_msg})

    # ── SMALL TALK ─────────────────────────────────────────────────────────
    if is_small_talk(user_msg):
        answer = "Hello! I am the InsightMed AI Assistant. How can I help you with your medical questions today?"
        history_store[session_id].append({"role": "assistant", "content": answer})
        return {"response": answer}

    # ── RAG PIPELINE ───────────────────────────────────────────────────────
    if qa_chain:
        try:
            answer = qa_chain.invoke(user_msg)

            if any(p in answer.lower() for p in ["not available", "ai model", "i don't know", "is not available", "does not contain"]):
                if llm_general:
                    # Pass the full conversation history to the LLM
                    messages = build_llm_messages(session_id, user_msg)
                    answer = llm_general.invoke(messages).content
        except Exception as e:
            answer = f"Error querying knowledge base: {str(e)}"

    elif llm_general:
        # No RAG chain — go straight to the general LLM with full history
        messages = build_llm_messages(session_id, user_msg)
        answer = llm_general.invoke(messages).content

    else:
        answer = "Language model is not initialized. Please check the server."

    history_store[session_id].append({"role": "assistant", "content": answer})
    return {"response": answer}


@app.get("/history")
async def get_history(session_id: str = "default_user"):
    return {"history": history_store.get(session_id, [])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)