from RAG_Chatbot.components.split.split import Split
from RAG_Chatbot.logging.logger import logging

split = Split()

all_chunks = []
for chunk in split.split_data():      # ← this triggers the whole chain
    all_chunks.append(chunk)

logging.info(f"Pipeline complete: {len(all_chunks)} total chunks")