import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv

from RAG_Chatbot.components.retriever.retriever import QAchain
from RAG_Chatbot.components.LLM.LLM import LLMs
from RAG_Chatbot.backend.action import book_table

load_dotenv()

app = FastAPI(title="Qiro Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    qa_chain = QAchain().retriever()
    llm_general = LLMs().llms_model()
except Exception as e:
    print("WARNING: Could not initialize QAchain/LLM on startup.", e)
    qa_chain = None
    llm_general = None

# Stores the full chat history per session
history_store: Dict[str, List[Dict[str, str]]] = {}

# Tracks in-progress bookings (multi-turn collection)
pending_bookings: Dict[str, dict] = {}

# FIX #2: Stores completed bookings so the bot can recall them later
completed_bookings: Dict[str, List[dict]] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default_user"


class ActionRequest(BaseModel):
    action_type: str
    parameters: dict
    session_id: str = "default_user"


def is_small_talk(message: str) -> bool:
    greetings = ["hi", "hello", "hey", "good morning", "how are you"]
    return message.lower().strip() in greetings


# Phrases that clearly signal the user wants to CREATE a new booking.
# Deliberately specific so that "tell me the booking" never matches.
_NEW_BOOKING_PHRASES = [
    "book a table", "book a seat", "book a spot",
    "make a reservation", "make a booking",
    "i want to book", "i'd like to book", "i would like to book",
    "can i book", "can i reserve", "can i make a reservation",
    "reserve a table", "reserve a seat", "i want to reserve",
    "place a reservation", "set up a booking", "create a booking",
]

# Words that signal the user is QUERYING an existing booking, not making one.
_BOOKING_QUERY_WORDS = [
    "tell me", "show me", "find", "look up", "lookup",
    "check", "list", "see", "get", "what", "which", "view",
    "show", "details", "info", "information", "fetch", "retrieve",
]


def is_new_booking_intent(message: str) -> bool:
    """Returns True only for clear 'I want to create a booking' messages."""
    msg = message.lower()
    return any(phrase in msg for phrase in _NEW_BOOKING_PHRASES)


def is_booking_lookup(message: str) -> bool:
    """
    Returns True when the user is asking ABOUT an existing booking
    (e.g. 'tell me the booking made on the name Yash').
    This must NOT trigger the booking state machine.
    """
    msg = message.lower()
    has_booking_noun = any(w in msg for w in ["booking", "reservation", "booked"])
    has_query_word = any(w in msg for w in _BOOKING_QUERY_WORDS)
    return has_booking_noun and has_query_word


# FIX #1: Build a full message list (system + history + current user msg)
# that can be passed directly to llm_general.invoke().
def build_llm_messages(session_id: str, current_user_msg: str) -> list:
    """
    Constructs the message list for the LLM, including:
      - A system prompt that embeds any known user facts and past bookings
      - The full conversation history for this session
      - The current user message
    """
    # Inject known bookings into the system prompt so the LLM can answer
    # questions like "what did I book?" without needing a special code path.
    booking_context = ""
    if session_id in completed_bookings and completed_bookings[session_id]:
        lines = []
        for b in completed_bookings[session_id]:
            lines.append(
                f"  - Name: {b['name']}, Date/Time: {b['datetime']}, Guests: {b['people']}"
            )
        booking_context = "\n\nCompleted reservations for this user:\n" + "\n".join(lines)

    system_prompt = (
        "You are the Qiro Verse AI Assistant. "
        "Be helpful, friendly, and concise. "
        "If the user asks about their name, bookings, or anything they mentioned earlier, "
        "answer using the conversation history provided."
        + booking_context
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

    if is_booking_lookup(user_msg) and session_id not in pending_bookings:
        bookings = completed_bookings.get(session_id, [])
        if not bookings:
            answer = "I don't have any bookings recorded for this session yet."
        else:
            lines = [
                f"  - {b['name']} — {b['datetime']} for {b['people']} guest(s)"
                for b in bookings
            ]
            answer = "Here are the bookings I have on record:\n" + "\n".join(lines)
        history_store[session_id].append({"role": "assistant", "content": answer})
        return {"response": answer}

    # ── BOOKING STATE MACHINE (create a new booking) ──────────────────────
    if is_new_booking_intent(user_msg) or session_id in pending_bookings:

        # INIT: brand-new booking
        if session_id not in pending_bookings:
            pending_bookings[session_id] = {
                "name": None,
                "date": None,
                "time": None,
                "people": None,
                "waiting_for": "name",
            }
            answer = "Sure! What name should I put the booking under?"
            history_store[session_id].append({"role": "assistant", "content": answer})
            return {"response": answer}

        # CONTINUE: fill the field we're currently waiting for
        state = pending_bookings[session_id]
        field = state["waiting_for"]

        if field == "name":
            state["name"] = user_msg
            state["waiting_for"] = "date"
            answer = f"Got it, '{state['name']}'! For which date? (e.g., Tomorrow, May 20)"

        elif field == "date":
            state["date"] = user_msg
            state["waiting_for"] = "time"
            answer = "What time works for you? (e.g., 7:00 PM)"

        elif field == "time":
            state["time"] = user_msg
            state["waiting_for"] = "people"
            answer = "How many guests will be joining?"

        elif field == "people":
            digits = "".join(filter(str.isdigit, user_msg))
            if not digits:
                answer = "Sorry, I need a number. How many guests will be joining?"
                history_store[session_id].append({"role": "assistant", "content": answer})
                return {"response": answer}

            state["people"] = int(digits)
            datetime_str = f"{state['date']} at {state['time']}"

            result = book_table(state["name"], datetime_str, state["people"])

            # FIX #2: Save the completed booking so it can be recalled later
            if session_id not in completed_bookings:
                completed_bookings[session_id] = []
            completed_bookings[session_id].append(
                {
                    "name": state["name"],
                    "datetime": datetime_str,
                    "people": state["people"],
                }
            )

            del pending_bookings[session_id]
            answer = result

        else:
            del pending_bookings[session_id]
            answer = "Something went wrong with the booking. Please say 'book a table' to start again."

        history_store[session_id].append({"role": "assistant", "content": answer})
        return {"response": answer}

    # ── SMALL TALK ─────────────────────────────────────────────────────────
    if is_small_talk(user_msg):
        answer = "Hello! I am the Qiro Verse AI Assistant. How can I help you today?"
        history_store[session_id].append({"role": "assistant", "content": answer})
        return {"response": answer}

    # ── RAG PIPELINE ───────────────────────────────────────────────────────
    if qa_chain:
        try:
            result = qa_chain.invoke({"query": user_msg})
            answer = result.get("result", "I couldn't retrieve an answer.")

            if any(p in answer.lower() for p in ["not available", "ai model", "i don't know"]):
                if llm_general:
                    # FIX #1: Pass the full conversation history to the LLM
                    messages = build_llm_messages(session_id, user_msg)
                    answer = llm_general.invoke(messages).content
        except Exception as e:
            answer = f"Error querying knowledge base: {str(e)}"

    elif llm_general:
        # No RAG chain — go straight to the general LLM with full history
        # FIX #1: Pass full conversation history here too
        messages = build_llm_messages(session_id, user_msg)
        answer = llm_general.invoke(messages).content

    else:
        answer = "Language model is not initialized. Please check the server."

    history_store[session_id].append({"role": "assistant", "content": answer})
    return {"response": answer}


@app.post("/action")
async def execute_action(req: ActionRequest):
    action_type = req.action_type
    params = req.parameters
    session_id = req.session_id

    if action_type == "book_table":
        name = params.get("name", "Guest")
        date = params.get("date", "Today")
        people = params.get("people", 2)
        result = book_table(name, date, people)

        # FIX #2: Also persist bookings made via the /action endpoint
        if session_id not in completed_bookings:
            completed_bookings[session_id] = []
        completed_bookings[session_id].append(
            {"name": name, "datetime": date, "people": people}
        )

        if session_id in history_store:
            history_store[session_id].append(
                {"role": "assistant", "content": f"[Action Executed] {result}"}
            )
        return {"status": "success", "response": result}

    return {"status": "failed", "response": f"Unknown action: {action_type}"}


@app.get("/history")
async def get_history(session_id: str = "default_user"):
    return {"history": history_store.get(session_id, [])}


# FIX #2: New endpoint to inspect completed bookings
@app.get("/bookings")
async def get_bookings(session_id: str = "default_user"):
    return {"bookings": completed_bookings.get(session_id, [])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)