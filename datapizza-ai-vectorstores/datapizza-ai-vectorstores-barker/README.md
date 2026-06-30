# datapizza-ai-vectorstores-barker

[Barker](https://github.com/barker-labs/barker) vectorstore integration for the [datapizza-ai](https://github.com/datapizza-labs/datapizza-ai) framework.

Barker is an in-memory, multi-core vector database written in Rust. It implements HNSW approximate nearest-neighbour search and exposes an HTTP REST API.

## Installation

```bash
pip install datapizza-ai-vectorstores-barker
```

## Quick Start

Start Barker (requires a running instance on port 8080):

```bash
# via barker-lite binary
./barker-main
```

Use it in datapizza-ai:

```python
from datapizza.vectorstores.barker import BarkerVectorstore
from datapizza.type import Chunk, DenseEmbedding

store = BarkerVectorstore(host="localhost", port=8080)

chunk = Chunk(
    id="doc-1",
    text="Barker è un database in-memory scritto in Rust",
    metadata={"source": "doc.txt"},
    embeddings=[DenseEmbedding(name="dense", vector=[0.1, 0.2, 0.3, 0.4])],
)
store.add(chunk, collection_name="my_docs")

results = store.search(
    collection_name="my_docs",
    query_vector=[0.1, 0.2, 0.3, 0.4],
    k=5,
)
for r in results:
    print(r.text, r.metadata["_distance"])
```

## Use in a RAG pipeline

```python
from datapizza.clients.anthropic import AnthropicClient
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.pipeline import DagPipeline
from datapizza.vectorstores.barker import BarkerVectorstore

client = AnthropicClient(api_key="...", model="claude-3-5-sonnet-latest")
embedder = OpenAIEmbedder(api_key="...", model_name="text-embedding-3-small")
store = BarkerVectorstore(host="localhost", port=8080)

dag = DagPipeline()
dag.add_module("embedder", embedder)
dag.add_module("retriever", store.as_retriever(collection_name="my_docs", k=5))
dag.add_module("prompt", ChatPromptTemplate(
    user_prompt_template="Domanda: {{user_prompt}}",
    retrieval_prompt_template="{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"
))
dag.add_module("generator", client)

dag.connect("embedder", "retriever", target_key="query_vector")
dag.connect("retriever", "prompt", target_key="chunks")
dag.connect("prompt", "generator", target_key="memory")

result = dag.run({
    "embedder": {"text": "Cos'è Barker?"},
    "prompt": {"user_prompt": "Cos'è Barker?"},
    "retriever": {"collection_name": "my_docs"},
    "generator": {"input": "Cos'è Barker?"},
})
print(result["generator"].text)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `host` | `"localhost"` | Barker server host |
| `port` | `8080` | Barker HTTP port |
| `api_key` | `None` | Bearer token for authenticated instances |
| `timeout` | `10.0` | HTTP request timeout in seconds |
| `metric` | `"COSINE"` | Distance metric: `"COSINE"`, `"EUCLIDEAN"`, `"DOT"` |

## How it works

Barker has a single global vector namespace. This integration uses key prefixes to emulate collections:

- Vectors are stored at `{collection_name}:vec:{chunk_id}`
- Metadata (text + chunk metadata) is stored at `{collection_name}:meta:{chunk_id}`

After a search, results are filtered by collection prefix and metadata is fetched for each matching chunk.
