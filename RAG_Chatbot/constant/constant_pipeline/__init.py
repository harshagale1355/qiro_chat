CSV_FILE_PATH = "/Qiro_chat/RAG_Chatbot/data/raw/data.csv"
PDF_FILE_PATH="/Qiro_chat/RAG_Chatbot/data/raw/data.pdf"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
MODEL_KWARG={'device': 'cpu'}
LLM_MODEL = "llama-3.1-8b-instant"
LLM_BASE_URL = "https://openrouter.ai/api/v1"
TEMPERATURE=0.1
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

SEPARATORS=[
    "\n\n",
    "\n",
    " ",
    ".",
    ",",
]

FILES=[
    "/home/harshagale/Documents/Projects/Qiro_chat/RAG_Chatbot/data/raw/cardio.pdf",
    "/home/harshagale/Documents/Projects/Qiro_chat/RAG_Chatbot/data/raw/diabetes.pdf",
    "/home/harshagale/Documents/Projects/Qiro_chat/RAG_Chatbot/data/raw/hypertension.pdf",
    "/home/harshagale/Documents/Projects/Qiro_chat/RAG_Chatbot/data/raw/tropical.pdf",
]
