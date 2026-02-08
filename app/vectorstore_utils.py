import os
import uuid
from tqdm import tqdm
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from sentence_transformers import SentenceTransformer


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "eurontest"

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
VECTOR_SIZE = model.get_sentence_embedding_dimension()


def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=True,
        timeout=180
    )


def reset_collection(client):

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )


# ✅ Store page metadata
def create_qdrant_index(chunks, batch_size=50):

    client = get_qdrant_client()
    reset_collection(client)

    points_batch = []

    for chunk in tqdm(chunks):

        # Extract text + page
        text = chunk.page_content
        page = chunk.metadata.get("page", "unknown")

        if len(text.strip()) < 5:
            continue

        vector = model.encode(text).tolist()

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                "page": page
            }
        )

        points_batch.append(point)

        if len(points_batch) >= batch_size:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )
            points_batch = []

    if points_batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_batch
        )

    return client


# ✅ Retrieval returns text + page
def retrive_similar_documents(client, query, k=5):

    query_vector = model.encode(str(query)).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k
    )

    docs = []
    for r in results.points:
        docs.append(
            f"(Page {r.payload['page']}) {r.payload['text']}"
        )

    return docs
