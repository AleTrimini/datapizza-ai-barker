import json
import uuid

import pytest
import respx
import httpx

from datapizza.type import Chunk, DenseEmbedding
from datapizza.vectorstores.barker import BarkerVectorstore

BASE = "http://localhost:8080"
COLLECTION = "test"
DIM = 4


def make_chunk(text: str = "hello world") -> Chunk:
    return Chunk(
        id=str(uuid.uuid4()),
        text=text,
        metadata={"source": "test.txt"},
        embeddings=[DenseEmbedding(name="dense", vector=[0.1, 0.2, 0.3, 0.4])],
    )


@respx.mock
def test_add_chunk():
    chunk = make_chunk()
    vec_key = f"{COLLECTION}:vec:{chunk.id}"
    meta_key = f"{COLLECTION}:meta:{chunk.id}"

    respx.post(f"{BASE}/api/v1/vectors/{vec_key}").mock(return_value=httpx.Response(200, json={"key": vec_key, "dim": DIM}))
    respx.post(f"{BASE}/api/v1/records/{meta_key}").mock(return_value=httpx.Response(200, json={}))

    store = BarkerVectorstore()
    store.add(chunk, collection_name=COLLECTION)


@respx.mock
def test_search_returns_chunks():
    chunk = make_chunk("il framework datapizza è scritto in Python")
    chunk_id = str(chunk.id)
    vec_key = f"{COLLECTION}:vec:{chunk_id}"
    meta_key = f"{COLLECTION}:meta:{chunk_id}"

    respx.post(f"{BASE}/api/v1/vectors/search").mock(return_value=httpx.Response(200, json={
        "results": [{"key": vec_key, "distance": 0.95}],
        "metric": "COSINE",
    }))
    respx.get(f"{BASE}/api/v1/records/{meta_key}").mock(return_value=httpx.Response(200, json={
        "value": {"text": chunk.text, "source": "test.txt"}
    }))

    store = BarkerVectorstore()
    results = store.search(collection_name=COLLECTION, query_vector=[0.1, 0.2, 0.3, 0.4], k=5)

    assert len(results) == 1
    assert results[0].text == chunk.text
    assert results[0].metadata["_distance"] == 0.95


@respx.mock
def test_search_filters_other_collections():
    """Risultati di altre collection non devono apparire."""
    respx.post(f"{BASE}/api/v1/vectors/search").mock(return_value=httpx.Response(200, json={
        "results": [{"key": "other_collection:vec:abc123", "distance": 0.99}],
        "metric": "COSINE",
    }))

    store = BarkerVectorstore()
    results = store.search(collection_name=COLLECTION, query_vector=[0.1, 0.2, 0.3, 0.4], k=5)
    assert results == []


@respx.mock
def test_retrieve_chunk():
    chunk_id = str(uuid.uuid4())
    meta_key = f"{COLLECTION}:meta:{chunk_id}"

    respx.get(f"{BASE}/api/v1/records/{meta_key}").mock(return_value=httpx.Response(200, json={
        "value": {"text": "testo recuperato", "source": "doc.txt"}
    }))

    store = BarkerVectorstore()
    results = store.retrieve(collection_name=COLLECTION, ids=[chunk_id])

    assert len(results) == 1
    assert results[0].text == "testo recuperato"
    assert results[0].metadata["source"] == "doc.txt"


@respx.mock
def test_remove_chunk():
    chunk_id = str(uuid.uuid4())
    vec_key = f"{COLLECTION}:vec:{chunk_id}"
    meta_key = f"{COLLECTION}:meta:{chunk_id}"

    respx.delete(f"{BASE}/api/v1/vectors/{vec_key}").mock(return_value=httpx.Response(204))
    respx.delete(f"{BASE}/api/v1/records/{meta_key}").mock(return_value=httpx.Response(204))

    store = BarkerVectorstore()
    store.remove(collection_name=COLLECTION, ids=[chunk_id])


@respx.mock
def test_health():
    respx.get(f"{BASE}/api/v1/health").mock(return_value=httpx.Response(200, json={
        "status": "ok", "version": "0.1.0", "uptime_seconds": 42.0,
        "total_keys": 10, "memory_mb": 1.5, "shards": 4,
        "vector_count": 5, "scache_count": 0,
    }))

    store = BarkerVectorstore()
    health = store.health()
    assert health["status"] == "ok"


@respx.mock
def test_add_chunk_missing_dense_embedding_raises():
    chunk = Chunk(id=str(uuid.uuid4()), text="no vector", metadata={}, embeddings=[])
    store = BarkerVectorstore()
    with pytest.raises(ValueError, match="no embeddings"):
        store.add(chunk, collection_name=COLLECTION)
