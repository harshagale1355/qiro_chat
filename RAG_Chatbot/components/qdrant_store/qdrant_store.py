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

class QdrantStore:

    def __init__(self):

        logging.info("Loading embedding model...")

        self.model = SentenceTransformer(MODEL_PATH)

        logging.info("Loading sparse model...")

        self.sparse_model = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME
        )

        logging.info("Loading splitter...")

        self.splitter = Split()

        logging.info("Connecting to Qdrant...")

        self.client = QdrantClient(
            path=QDRANT_PATH
        )

    # -------------------------------------------------------------------------
    # Create Collection
    # -------------------------------------------------------------------------

    def create_collection(self):

        collections = self.client.get_collections().collections

        collection_names = [
            collection.name
            for collection in collections
        ]

        if COLLECTION_NAME in collection_names:

            logging.info(
                f"Collection '{COLLECTION_NAME}' already exists."
            )

            return

        logging.info(
            f"Creating collection: {COLLECTION_NAME}"
        )

        self.client.create_collection(

            collection_name=COLLECTION_NAME,

            vectors_config={
                "dense": models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },

            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(
                        on_disk=False,
                    )
                )
            }
        )

    # -------------------------------------------------------------------------
    # Existing Sources
    # -------------------------------------------------------------------------

    def get_existing_sources(self):

        existing_sources = set()

        scroll_result = self.client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100000,
            with_payload=True,
            with_vectors=False,
        )

        points = scroll_result[0]

        for point in points:

            payload = point.payload or {}

            source = payload.get("source")

            if source:
                existing_sources.add(source)

        return existing_sources

    # -------------------------------------------------------------------------
    # Store Documents
    # -------------------------------------------------------------------------

    def store(self, reset: bool = False):

        if reset:

            self.client.delete_collection(
                COLLECTION_NAME
            )

        self.create_collection()

        existing_sources = self.get_existing_sources()

        all_chunks = list(
            self.splitter.split_data()
        )

        new_chunks = [

            chunk

            for chunk in all_chunks

            if chunk.metadata.get("source")
            not in existing_sources
        ]

        if not new_chunks:

            logging.info(
                "No new documents found."
            )

            return

        texts = [
            chunk.page_content
            for chunk in new_chunks
        ]

        # Dense Embeddings
        dense_embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Sparse Embeddings
        sparse_embeddings = list(
            self.sparse_model.embed(texts)
        )

        points = []

        for chunk, dense_vec, sparse_vec in zip(
            new_chunks,
            dense_embeddings,
            sparse_embeddings
        ):

            points.append(

                PointStruct(

                    id=str(uuid.uuid4()),

                    vector={

                        "dense": dense_vec.tolist(),

                        "sparse": sparse_vec.as_object()
                    },

                    payload={

                        "text": chunk.page_content,

                        "source": chunk.metadata.get(
                            "source",
                            "unknown"
                        ),

                        "topic": chunk.metadata.get(
                            "topic",
                            "General"
                        ),

                        "page": chunk.metadata.get(
                            "page",
                            -1
                        ),
                    }
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        logging.info(
            f"Stored {len(points)} chunks in Qdrant."
        )

    # -------------------------------------------------------------------------
    # Close
    # -------------------------------------------------------------------------

    def close(self):

        self.client.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    store = QdrantStore()

    store.store(reset=True)

    store.close()