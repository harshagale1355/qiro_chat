from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from RAG_Chatbot.components.split.split import Split
from RAG_Chatbot.logging.logger import logging

import uuid


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COLLECTION_NAME = "medical_rag_collection"

MODEL_PATH = "RAG_Chatbot/components/fine_tune_embed/fine_tuned_model"

QDRANT_PATH = "./qdrant_storage"

SPARSE_MODEL_NAME = "Qdrant/bm25"

BATCH_SIZE = 32

VECTOR_SIZE = 384


# -----------------------------------------------------------------------------
# Qdrant Store
# -----------------------------------------------------------------------------

import uuid
import sys
from typing import List, Dict, Any
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langsmith import traceable



class QdrantStore:
    def __init__(self):
        logging.info("Initializing QdrantStore...")
        self.model = SentenceTransformer(MODEL_PATH)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        self.splitter = Split()  # This now returns an LCEL generator
        self.client = QdrantClient(path=QDRANT_PATH)

    @traceable(run_type="embedding")
    def _generate_embeddings(self, texts: List[str]):
        """Traceable embedding generation for LangSmith."""
        dense = self.model.encode(
            texts, 
            batch_size=BATCH_SIZE, 
            convert_to_numpy=True, 
            normalize_embeddings=True
        )
        sparse = list(self.sparse_model.embed(texts))
        return dense, sparse

    def create_collection(self):
        # ... (Keep your existing create_collection logic) ...
        pass

    def get_existing_sources(self):
        # ... (Keep your existing get_existing_sources logic) ...
        pass

    def _ingest_documents(self, input_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """The core logic wrapped for LCEL."""
        try:
            reset = (input_data or {}).get("reset", False)
            if reset:
                self.client.delete_collection(COLLECTION_NAME)
            
            self.create_collection()
            existing_sources = self.get_existing_sources()

            # Using the LCEL splitter from the previous step
            all_chunks = list(self.splitter.split_data())
            new_chunks = [c for c in all_chunks if c.metadata.get("source") not in existing_sources]

            if not new_chunks:
                return {"status": "success", "chunks_stored": 0}

            texts = [chunk.page_content for chunk in new_chunks]
            
            # Traceable embedding call
            dense_embeddings, sparse_embeddings = self._generate_embeddings(texts)

            points = []
            for chunk, dense_vec, sparse_vec in zip(new_chunks, dense_embeddings, sparse_embeddings):
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector={"dense": dense_vec.tolist(), "sparse": sparse_vec.as_object()},
                        payload={
                            "text": chunk.page_content,
                            **chunk.metadata # Efficiently unpack metadata
                        }
                    )
                )

            self.client.upsert(collection_name=COLLECTION_NAME, points=points)
            logging.info(f"Stored {len(points)} chunks in Qdrant.")
            
            return {"status": "success", "chunks_stored": len(points)}

        except Exception as e:
            from RAG_Chatbot.exception.exception import RAG_Chatbot_Exception
            raise RAG_Chatbot_Exception(e, sys)

    def as_runnable(self):
        """Expose the store logic as an LCEL Runnable."""
        return RunnableLambda(self._ingest_documents)

# -----------------------------------------------------------------------------
# Main Implementation with LCEL
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    store_component = QdrantStore()
    
    # Define the Ingestion Chain
    # In a full LCEL flow, you could pipe your loader | splitter | store
    ingestion_chain = store_component.as_runnable()
    
    # Run the chain
    result = ingestion_chain.invoke({"reset": True})
    print(f"Ingestion Result: {result}")
    
    store_component.close()