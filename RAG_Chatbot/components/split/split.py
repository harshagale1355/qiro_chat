import sys
from typing import Iterator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
from RAG_Chatbot.constant.constant_pipeline.__init import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS
from RAG_Chatbot.components.data_load.data_loader import Documentloader

class Splitter:
    def __init__(self):
        # Initialize the components
        self.loader = Documentloader()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS,
        )

    def _transform(self, input_data=None) -> Iterator[Document]:
        """Core logic for splitting documents, wrapped for LCEL."""
        try:
            # document_loader is now part of the sequence
            data = self.loader.document_loader()
            total_chunks = 0

            for doc in data:
                chunks = self.text_splitter.split_documents([doc])
                for chunk in chunks:
                    total_chunks += 1
                    yield chunk

            logging.info(f"Splitting complete. Total chunks: {total_chunks}")

        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)

    def get_splitter_chain(self):
        return RunnableLambda(self._transform)

