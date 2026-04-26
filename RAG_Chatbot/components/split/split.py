from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
from RAG_Chatbot.constant.constant_pipeline.__init import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS
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
            # Load all PDFs as Document objects
            data = loader.document_loader()
            
            # Split into chunks (keeps metadata like file and page info)
            splits = text_splitter.split_documents(data)
            
            logging.info(f"Data splitting complete, {len(splits)} chunks created!")
            return splits
        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)