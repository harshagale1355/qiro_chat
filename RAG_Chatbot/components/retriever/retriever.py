

COLLECTION_NAME = "medical_rag_collection"

QDRANT_PATH = "./qdrant_storage"

MODEL_PATH = "RAG_Chatbot/components/fine_tune_embed/fine_tuned_model"

SPARSE_MODEL_NAME = "Qdrant/bm25"


from typing import List
import sys

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr
from langsmith import traceable  # Added for granular tracing

from RAG_Chatbot.logging.logger import logging

# ... (Configuration constants remain the same) ...

class QdrantRetriever(BaseRetriever):
    limit: int = 1
    _client: QdrantClient = PrivateAttr()
    _model: SentenceTransformer = PrivateAttr()
    _sparse_model: SparseTextEmbedding = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logging.info("Initializing QdrantRetriever...")
        self._model = SentenceTransformer(MODEL_PATH)
        self._sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
        self._client = QdrantClient(path=QDRANT_PATH)

    @traceable(run_type="embedding", name="Hybrid Encoding")
    def _encode_query(self, query: str):
        """Separate method for tracing embedding generation in LangSmith."""
        dense = self._model.encode(query, normalize_embeddings=True).tolist()
        sparse = list(self._sparse_model.embed([query]))[0]
        return dense, sparse

    @traceable(run_type="retriever", name="Qdrant Hybrid Search")
    def _get_relevant_documents(self, query: str) -> List[Document]:
        logging.info(f"Retriever triggered for query: {query}")

        # 1. Generate Embeddings (Traced as a sub-step)
        dense_query, sparse_query = self._encode_query(query)

        # 2. Hybrid Search in Qdrant
        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=self.limit * 5, # Fetch extra for RRF fusion
                ),
                models.Prefetch(
                    query=models.SparseVector(**sparse_query.as_object()),
                    using="sparse",
                    limit=self.limit * 5,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=self.limit,
        )

        # 3. Process results into LangChain Documents
        documents = []
        for point in results.points:
            payload = point.payload or {}
            doc = Document(
                page_content=payload.get("text", ""),
                metadata={
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "score": point.score,
                    "retrieval_type": "hybrid_rrf"
                }
            )
            documents.append(doc)

        return documents

    def close(self):
        self._client.close()
