from langchain_community.document_loaders import PyPDFLoader
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.constant.constant_pipeline.__init import FILES
import os, sys

class Documentloader:
    """Loads all PDFs specified in FILES into a list of LangChain Documents."""

    def document_loader(self):
        all_data = []
        try:
            if not FILES:
                logging.warning("No files specified in FILES!")
                return []
            
            for file in FILES:
                if not os.path.exists(file):
                    logging.warning(f"{file} not found, skipping...")
                    continue
                
                loader = PyPDFLoader(file_path=file)
                docs = loader.load()
                all_data.extend(docs)
                logging.info(f"{file} loaded successfully ({len(docs)} pages)")
            
            return all_data
        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)