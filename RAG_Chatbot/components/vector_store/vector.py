from RAG_Chatbot.logging.logger import logging
from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception

from langchain_text_splitters import RecursiveCharacterTextSplitter
from RAG_Chatbot.components.split.split import Split
from RAG_Chatbot.components.embedding.embedding import embedding
import sys
from langchain_community.vectorstores import Chroma

class VectorStore:
    def vector(self):
        try:
            split = Split()
            splitter_data = split.split_data()

            embed_instance = embedding()  # instantiate the class
            embedding_doc = embed_instance.embedding_data()  # call the method properly

            vector_store = Chroma(
                embedding_function=embedding_doc,
                persist_directory="./chroma_db"
            )
            for chunk in splitter_data:
                vector_store.add_documents([chunk])

            vector_store.persist()
            logging.info("Vector is stored")
            return vector_store
        except Exception as e:
            raise RAG_Chatbot_Exception(e, sys)