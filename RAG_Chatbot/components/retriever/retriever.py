from typing import List

from qdrant_client import QdrantClient
from qdrant_client.http import models

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from pydantic import PrivateAttr

from RAG_Chatbot.logging.logger import logging


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

COLLECTION_NAME = "medical_rag_collection"

QDRANT_PATH = "./qdrant_storage"

MODEL_PATH = "RAG_Chatbot/components/fine_tune_embed/fine_tuned_model"

SPARSE_MODEL_NAME = "Qdrant/bm25"


# -----------------------------------------------------------------------------
# Retriever
# -----------------------------------------------------------------------------

class QdrantRetriever(BaseRetriever):

    limit: int = 1

    # Private attributes
    _client: QdrantClient = PrivateAttr()

    _model: SentenceTransformer = PrivateAttr()

    _sparse_model: SparseTextEmbedding = PrivateAttr()

    # -------------------------------------------------------------------------
    # Init
    # -------------------------------------------------------------------------

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        logging.info("Loading embedding model...")

        self._model = SentenceTransformer(
            MODEL_PATH
        )

        logging.info("Loading sparse model...")

        self._sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME
        )

        logging.info("Connecting to Qdrant...")

        self._client = QdrantClient(
            path=QDRANT_PATH
        )

    # -------------------------------------------------------------------------
    # Required LangChain Method
    # -------------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str
    ) -> List[Document]:

        logging.info(f"Searching: {query}")

        # -------------------------------------------------------------------------
        # Dense Embedding
        # -------------------------------------------------------------------------

        dense_query = self._model.encode(
            query,
            normalize_embeddings=True
        ).tolist()

        # -------------------------------------------------------------------------
        # Sparse Embedding
        # -------------------------------------------------------------------------

        sparse_query = list(
            self._sparse_model.embed([query])
        )[0]

        # -------------------------------------------------------------------------
        # Hybrid Search
        # -------------------------------------------------------------------------

        results = self._client.query_points(

            collection_name=COLLECTION_NAME,

            prefetch=[

                # Dense Search
                models.Prefetch(
                    query=dense_query,
                    using="dense",

                    # Fetch more candidates
                    limit=1,
                ),

                # Sparse Search
                models.Prefetch(
                    query=models.SparseVector(
                        **sparse_query.as_object()
                    ),

                    using="sparse",

                    # Fetch more candidates
                    limit=1,
                ),
            ],

            # Fusion
            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),

            # Final top-k
            limit=self.limit,
        )

        documents = []

        # -------------------------------------------------------------------------
        # Convert to LangChain Documents
        # -------------------------------------------------------------------------

        for index, point in enumerate(results.points):

            payload = point.payload or {}

            doc = Document(

                page_content=payload.get(
                    "text",
                    ""
                ),

                metadata={

                    "source": payload.get(
                        "source"
                    ),

                    "topic": payload.get(
                        "topic"
                    ),

                    "page": payload.get(
                        "page"
                    ),

                    "score": point.score,
                }
            )

            documents.append(doc)

            # ---------------------------------------------------------------------
            # Debug Print
            # ---------------------------------------------------------------------

            print("\n" + "=" * 80)

            print(f"RESULT {index + 1}")

            print(f"Score: {point.score}")

            print(f"Page: {payload.get('page')}")

            print(doc.page_content[:1000])

        logging.info(
            f"Retrieved {len(documents)} documents"
        )

        return documents

    # -------------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------------

    def close(self):

        self._client.close()