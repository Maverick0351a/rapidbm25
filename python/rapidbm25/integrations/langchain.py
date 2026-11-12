"""LangChain retriever backed by :class:`rapidbm25.LiveIndex`.

Install with ``pip install rapidbm25[langchain]`` (``langchain-core``).
``from_texts`` / ``from_documents`` build the index, ``invoke`` returns
``Document`` objects, and ``save`` / ``load`` round-trip scores through the
versioned on-disk format.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict, Field, PrivateAttr
except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
    raise ImportError(
        "rapidbm25.integrations.langchain needs langchain-core. Install it with: pip install rapidbm25[langchain]"
    ) from exc

from rapidbm25.live import LiveIndex


def _build_live(texts, metadatas, ids, variant, bm25_params) -> LiveIndex:
    params = dict(bm25_params or {})
    live = LiveIndex(variant=variant, **params)
    metas = list(metadatas) if metadatas is not None else [{} for _ in texts]
    if len(metas) != len(texts):
        raise ValueError("metadatas must have the same length as texts")
    if ids is None:
        ids = [str(i) for i in range(len(texts))]
    if len(ids) != len(texts):
        raise ValueError("ids must have the same length as texts")
    for doc_id, text, meta in zip(ids, texts, metas):
        live.add(str(doc_id), text, metadata=dict(meta or {}))
    return live


class BM25Retriever(BaseRetriever):
    """BM25 retriever. ``k`` is the number of documents ``invoke`` returns."""

    k: int = 4
    variant: str = "okapi"
    texts: list[str] = Field(default_factory=list)
    metadatas: Optional[list[dict]] = None
    ids: Optional[list[str]] = None
    bm25_params: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    _live: Any = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self._live is None:
            self._live = _build_live(self.texts, self.metadatas, self.ids, self.variant, self.bm25_params)

    @classmethod
    def from_texts(cls, texts: Sequence[str], metadatas: Optional[Sequence[dict]] = None, *,
                   ids: Optional[Sequence[str]] = None, k: int = 4, variant: str = "okapi", **kwargs):
        """Build from raw strings. BM25 parameters (``k1``, ``b``, ``delta``, ``compat``, ``fixes``, ``tokenizer``)
        may be passed directly or inside ``bm25_params``."""
        params = dict(kwargs.pop("bm25_params", {}) or {})
        for key in ("k1", "b", "epsilon", "delta", "compat", "fixes", "tokenizer"):
            if key in kwargs:
                params[key] = kwargs.pop(key)
        return cls(
            k=k,
            texts=list(texts),
            metadatas=None if metadatas is None else [dict(m) for m in metadatas],
            ids=None if ids is None else [str(i) for i in ids],
            variant=variant,
            bm25_params=params,
            **kwargs,
        )

    @classmethod
    def from_documents(cls, documents: Sequence[Document], **kwargs):
        texts = [d.page_content for d in documents]
        metas = [dict(d.metadata or {}) for d in documents]
        ids = [str(d.id) if d.id is not None else str(i) for i, d in enumerate(documents)]
        return cls.from_texts(texts, metadatas=metas, ids=ids, **kwargs)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        del run_manager
        hits = self._live.search(query, k=self.k)
        docs = []
        for hit in hits:
            meta = dict(hit["metadata"])
            meta["bm25_score"] = hit["score"]
            docs.append(Document(id=hit["id"], page_content=hit["text"], metadata=meta))
        return docs

    def save(self, path: str) -> None:
        """Write a directory containing the live index and a small JSON sidecar."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self._live.save(root / "index.rbm25")
        sidecar = {
            "k": self.k,
            "variant": self.variant,
            "bm25_params": {key: value for key, value in self.bm25_params.items() if key != "tokenizer"},
        }
        (root / "retriever.json").write_text(json.dumps(sidecar), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "BM25Retriever":
        root = Path(path)
        sidecar = json.loads((root / "retriever.json").read_text(encoding="utf-8"))
        live = LiveIndex.load(root / "index.rbm25")
        obj = cls.model_construct(
            k=int(sidecar.get("k", 4)),
            variant=sidecar.get("variant", live.variant),
            texts=live.texts(),
            metadatas=live.metadatas(),
            ids=live.ids(),
            bm25_params=dict(sidecar.get("bm25_params") or {}),
        )
        obj._live = live
        return obj
