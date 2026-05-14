from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
from RAG_Chatbot.constant.constant_pipeline.__init import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS, FILES
from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG_Chatbot.components.data_load.data_loader import Documentloader
import sys

class Split:
    def split_data(self):
        try:
            loader = Documentloader()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                separators=SEPARATORS,
            )

            data = loader.document_loader()
            total_chunks = 0

            for doc in data:
                chunks = text_splitter.split_documents([doc])
                for chunk in chunks:
                    total_chunks += 1
                    yield chunk

            logging.info(f"Splitting complete. Total chunks: {total_chunks}")  # ✅ after loop, before generator exhausts

        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)