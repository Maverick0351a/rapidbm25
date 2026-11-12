"""LlamaIndex retriever backed by :class:`rapidbm25.LiveIndex`.

Install with ``pip install rapidbm25[llama-index]`` (``llama-index-core``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
    raise ImportError(
        "rapidbm25.integrations.llama_index needs llama-index-core. "
        "Install it with: pip install rapidbm25[llama-index]"
    ) from exc

from rapidbm25.live import LiveIndex


class BM25Retriever(BaseRetriever):
    """Lexical retriever. ``retrieve`` returns ``NodeWithScore`` nodes."""

    def __init__(self, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None, *,
                 ids: Optional[Sequence[str]] = None, k: int = 2, variant: str = "okapi",
                 bm25_params: Optional[dict] = None, **kwargs: Any):
        super().__init__(**{key: value for key, value in kwargs.items() if key in ("callback_manager", "verbose")})
        self._k = int(k)
        self._variant = variant
        params = dict(bm25_params or {})
        self._live = LiveIndex(variant=variant, **params)
        metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
        if ids is None:
            ids = [str(i) for i in range(len(texts))]
        for doc_id, text, meta in zip(ids, texts, metas):
            self._live.add(str(doc_id), text, metadata=dict(meta or {}))

    @classmethod
    def from_texts(cls, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None, **kwargs):
        return cls(texts, metadatas=metadatas, **kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query = query_bundle.query_str
        hits = self._live.search_terms(query, k=self._k)
        nodes = []
        for hit in hits:
            node = TextNode(id_=hit["id"], text=hit["text"], metadata=dict(hit["metadata"]))
            nodes.append(NodeWithScore(node=node, score=float(hit["score"])))
        return nodes

    def save(self, path: str) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self._live.save(root / "index.rbm25")
        (root / "retriever.json").write_text(json.dumps({"k": self._k, "variant": self._variant}), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        root = Path(path)
        sidecar = json.loads((root / "retriever.json").read_text(encoding="utf-8"))
        live = LiveIndex.load(root / "index.rbm25")
        obj = cls.__new__(cls)
        BaseRetriever.__init__(obj)
        obj._k = int(sidecar.get("k", 2))
        obj._variant = sidecar.get("variant", live.variant)
        obj._live = live
        return obj
