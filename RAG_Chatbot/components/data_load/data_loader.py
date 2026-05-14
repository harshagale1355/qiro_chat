from langchain_community.document_loaders import PyPDFLoader
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.constant.constant_pipeline.__init import FILES
import os, sys

class Documentloader:
    """Loads all PDFs specified in FILES into a list of LangChain Documents."""

    def document_loader(self):
        try:
            if not FILES:
                logging.warning("No files specified in FILES!")
                return
            
             
            # Open the file once before processing all PDFs

            for file in FILES:
                if not os.path.exists(file):
                    logging.warning(f"{file} not found, skipping...")
                    continue

                loader = PyPDFLoader(file_path=file)
                docs = loader.load()

                logging.info(f"Loaded {len(docs)} pages from: {file}")

                for doc in docs:
                    yield doc

        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)