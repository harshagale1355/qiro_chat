import os
import sys
from typing import Iterator
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.constant.constant_pipeline.__init import FILES

class Documentloader:
    """Loads PDFs using LCEL compatible logic for better LangSmith tracing."""

    def __init__(self):
        self.files = FILES

    def _process_files(self, config: dict = None) -> Iterator[Document]:
        """The core logic wrapped for LCEL."""
        try:
            if not self.files:
                logging.warning("No files specified in FILES!")
                return

            for file_path in self.files:
                if not os.path.exists(file_path):
                    logging.warning(f"{file_path} not found, skipping...")
                    continue

                # LangSmith will now track this specific loader activity
                loader = PyPDFLoader(file_path=file_path)
                docs = loader.load()

                logging.info(f"Loaded {len(docs)} pages from: {file_path}")
                
                for doc in docs:
                    yield doc

        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)

    def get_loader_chain(self):
        return RunnableLambda(self._process_files)

