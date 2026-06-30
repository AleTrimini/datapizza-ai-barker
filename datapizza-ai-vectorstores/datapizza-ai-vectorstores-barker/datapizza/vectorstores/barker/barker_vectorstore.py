import logging
import uuid
from typing import Any
from urllib.parse import quote

try:
    import httpx
except ImportError as exc:
    raise ImportError(
        "httpx is required: pip install datapizza-ai-vectorstores-barker"
    ) from exc

from datapizza.core.vectorstore import Vectorstore
from datapizza.type import Chunk, DenseEmbedding

log = logging.getLogger(__name__)


class BarkerVectorstore(Vectorstore):
    """
    datapizza-ai integration for Barker — an in-memory, multi-core vector database written in Rust.

    Barker exposes vector operations via its HTTP REST API (/api/v1/vectors/*).
    Each chunk is stored as a vector keyed by its ID; metadata and text are
    stored alongside in the KV store (/api/v1/records/*).

    Args:
        host: Barker server host. Defaults to "localhost".
        port: Barker server HTTP port. Defaults to 8080.
        api_key: Optional Bearer token for authenticated Barker instances.
        timeout: HTTP request timeout in seconds. Defaults to 10.0.
        metric: Distance metric for vector search ("COSINE", "EUCLIDEAN", "DOT"). Defaults to "COSINE".
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        api_key: str | None = None,
        timeout: float = 10.0,
        metric: str = "COSINE",
    ):
        self.base_url = f"http://{host}:{port}"
        self.metric = metric
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers
        self._timeout = timeout

    # ── internal helpers ──────────────────────────────────────────────

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url, headers=self._headers, timeout=self._timeout
        )

    def _a_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, headers=self._headers, timeout=self._timeout
        )

    def _req(self, method: str, path: str, json: Any = None, params: dict | None = None) -> Any:
        with self._client() as client:
            res = client.request(method, path, json=json, params=params)
        if res.status_code == 204:
            return None
        data = res.json()
        if not res.is_success:
            raise RuntimeError(f"Barker error {res.status_code}: {data.get('error', res.text)}")
        return data

    async def _a_req(self, method: str, path: str, json: Any = None, params: dict | None = None) -> Any:
        async with self._a_client() as client:
            res = await client.request(method, path, json=json, params=params)
        if res.status_code == 204:
            return None
        data = res.json()
        if not res.is_success:
            raise RuntimeError(f"Barker error {res.status_code}: {data.get('error', res.text)}")
        return data

    def _chunk_to_vector_key(self, chunk_id: str) -> str:
        return f"vec:{chunk_id}"

    def _chunk_to_meta_key(self, chunk_id: str) -> str:
        return f"meta:{chunk_id}"

    def _extract_vector(self, chunk: Chunk) -> list[float]:
        if not chunk.embeddings:
            raise ValueError(f"Chunk {chunk.id} has no embeddings")
        for emb in chunk.embeddings:
            if isinstance(emb, DenseEmbedding):
                return emb.vector
        raise ValueError(f"Chunk {chunk.id} has no dense embedding — Barker only supports dense vectors")

    # ── Vectorstore interface ─────────────────────────────────────────

    def add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        """Store chunks in Barker. Each chunk is saved as a vector + a KV metadata record.

        Note: Barker has a single global vector namespace; collection_name is used
        as a key prefix to logically separate collections.
        """
        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        prefix = f"{collection_name}:" if collection_name else ""

        for c in chunks:
            chunk_id = str(c.id) if c.id else str(uuid.uuid4())
            vector = self._extract_vector(c)
            vec_key = prefix + self._chunk_to_vector_key(chunk_id)
            meta_key = prefix + self._chunk_to_meta_key(chunk_id)

            self._req("POST", f"/api/v1/vectors/{quote(vec_key, safe='')}", json={"vector": vector})
            self._req("POST", f"/api/v1/records/{quote(meta_key, safe='')}", json={"value": {"text": c.text, **c.metadata}})
            log.debug(f"Added chunk {chunk_id} to Barker (collection={collection_name})")

    async def a_add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        prefix = f"{collection_name}:" if collection_name else ""

        for c in chunks:
            chunk_id = str(c.id) if c.id else str(uuid.uuid4())
            vector = self._extract_vector(c)
            vec_key = prefix + self._chunk_to_vector_key(chunk_id)
            meta_key = prefix + self._chunk_to_meta_key(chunk_id)

            await self._a_req("POST", f"/api/v1/vectors/{quote(vec_key, safe='')}", json={"vector": vector})
            await self._a_req("POST", f"/api/v1/records/{quote(meta_key, safe='')}", json={"value": {"text": c.text, **c.metadata}})

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        metric: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """Search for the k nearest vectors in Barker using HNSW.

        Args:
            collection_name: Logical collection to search in (used as key prefix).
            query_vector: Dense query vector as list of floats.
            k: Number of results to return.
            metric: Distance metric override ("COSINE", "EUCLIDEAN", "DOT").
        """
        data = self._req(
            "POST",
            "/api/v1/vectors/search",
            json={"k": k, "vector": query_vector, "metric": metric or self.metric},
        )
        results = data.get("results", [])
        prefix = f"{collection_name}:" if collection_name else ""

        chunks = []
        for r in results:
            key: str = r.get("key", "")
            # Filter by collection prefix and only return vector keys
            if not key.startswith(prefix + "vec:"):
                continue
            chunk_id = key[len(prefix + "vec:"):]
            meta_key = prefix + self._chunk_to_meta_key(chunk_id)

            try:
                meta_data = self._req("GET", f"/api/v1/records/{quote(meta_key, safe='')}")
                payload = meta_data.get("value", {})
            except RuntimeError:
                log.warning(f"Metadata not found for chunk {chunk_id}, skipping")
                continue

            text = payload.pop("text", "")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata={**payload, "_distance": r.get("distance")},
                    embeddings=[DenseEmbedding(name="dense", vector=query_vector)],
                )
            )
        return chunks

    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        metric: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        data = await self._a_req(
            "POST",
            "/api/v1/vectors/search",
            json={"k": k, "vector": query_vector, "metric": metric or self.metric},
        )
        results = data.get("results", [])
        prefix = f"{collection_name}:" if collection_name else ""

        chunks = []
        for r in results:
            key: str = r.get("key", "")
            if not key.startswith(prefix + "vec:"):
                continue
            chunk_id = key[len(prefix + "vec:"):]
            meta_key = prefix + self._chunk_to_meta_key(chunk_id)

            try:
                meta_data = await self._a_req("GET", f"/api/v1/records/{quote(meta_key, safe='')}")
                payload = meta_data.get("value", {})
            except RuntimeError:
                log.warning(f"Metadata not found for chunk {chunk_id}, skipping")
                continue

            text = payload.pop("text", "")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    metadata={**payload, "_distance": r.get("distance")},
                    embeddings=[DenseEmbedding(name="dense", vector=query_vector)],
                )
            )
        return chunks

    def retrieve(self, collection_name: str, ids: list[str], **kwargs) -> list[Chunk]:
        """Retrieve chunks by ID from Barker's KV store."""
        prefix = f"{collection_name}:" if collection_name else ""
        chunks = []
        for chunk_id in ids:
            meta_key = prefix + self._chunk_to_meta_key(chunk_id)
            try:
                meta_data = self._req("GET", f"/api/v1/records/{quote(meta_key, safe='')}")
                payload = meta_data.get("value", {})
                text = payload.pop("text", "")
                chunks.append(Chunk(id=chunk_id, text=text, metadata=payload))
            except RuntimeError:
                log.warning(f"Chunk {chunk_id} not found in Barker")
        return chunks

    def update(self, collection_name: str, payload: dict, points: list[int], **kwargs):
        """Update metadata for existing chunks by overwriting their KV records."""
        prefix = f"{collection_name}:" if collection_name else ""
        for chunk_id in points:
            meta_key = prefix + self._chunk_to_meta_key(str(chunk_id))
            self._req("POST", f"/api/v1/records/{quote(meta_key, safe='')}", json={"value": payload})

    def remove(self, collection_name: str, ids: list[str], **kwargs):
        """Delete chunks (vector + metadata) from Barker."""
        prefix = f"{collection_name}:" if collection_name else ""
        for chunk_id in ids:
            vec_key = prefix + self._chunk_to_vector_key(chunk_id)
            meta_key = prefix + self._chunk_to_meta_key(chunk_id)
            self._req("DELETE", f"/api/v1/vectors/{quote(vec_key, safe='')}")
            self._req("DELETE", f"/api/v1/records/{quote(meta_key, safe='')}")

    def health(self) -> dict:
        """Check Barker server health."""
        return self._req("GET", "/api/v1/health")

    def reindex(self) -> dict:
        """Rebuild the HNSW index and remove tombstones."""
        return self._req("POST", "/api/v1/vectors/reindex")
