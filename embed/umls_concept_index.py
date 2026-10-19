# === embed/umls_concept_index.py ===
"""
逐 entity 的 UMLS concept linking。

取代舊作法「把整段 patient context 當一個 query 對 29 萬個 concept 做 FAISS」，
改成：對每一個臨床 entity 名稱（診斷 / 處置 / 藥名）各自連結到最合適的 CUI。

流程：
  1. 正規化字串精確比對（name -> cui），命中即 score = 1.0
  2. 未命中 → 用 sentence embedding 做語意最近鄰（concept 向量矩陣快取在磁碟）

concept 向量只會建立一次，之後所有病人共用。
"""
from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

import numpy as np

from embed.encoder import Encoder
from embed.faiss_index import FaissIndex
from retrieval.umls_client import UMLSClient

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize(text: str) -> str:
    text = (text or "").strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


class UMLSConceptIndex:
    """建立一次、跨病人共用的 UMLS concept 連結器。"""

    _shared: Dict[str, "UMLSConceptIndex"] = {}

    def __init__(self, umls_dir: str, encoder: Optional[Encoder] = None):
        self.umls_dir = umls_dir
        self.encoder = encoder or Encoder()
        self.model_name = getattr(self.encoder, "model_name", "encoder")

        client = UMLSClient(umls_dir)
        self.cuis: List[str] = list(client.concepts)
        self.names: List[str] = [client.concept_names[c] for c in self.cuis]

        # 正規化字串 -> cui（精確比對用；同名取第一個）
        self._exact: Dict[str, str] = {}
        for cui, name in zip(self.cuis, self.names):
            key = _normalize(name)
            if key and key not in self._exact:
                self._exact[key] = cui

        self._vecs: Optional[np.ndarray] = None
        self._index: Optional[FaissIndex] = None

    # ---- 分類方法 ---------------------------------------------------
    @classmethod
    def shared(cls, umls_dir: str, encoder: Optional[Encoder] = None) -> "UMLSConceptIndex":
        key = os.path.abspath(umls_dir)
        if key not in cls._shared:
            cls._shared[key] = cls(umls_dir, encoder=encoder)
        return cls._shared[key]

    # ---- concept 向量（磁碟快取）----------------------------------
    @property
    def _cache_path(self) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.model_name)
        return os.path.join(self.umls_dir, f"concept_vecs_{safe}.npy")

    def _ensure_index(self) -> None:
        if self._index is not None:
            return
        path = self._cache_path
        if os.path.exists(path):
            vecs = np.load(path).astype(np.float32)
            if vecs.shape[0] != len(self.names):
                vecs = None  # concept 清單變了 → 重建
        else:
            vecs = None

        if vecs is None:
            print(f"[UMLSConceptIndex] building concept embeddings for {len(self.names)} names "
                  f"(one-off, cached to {path}) ...")
            t0 = time.time()
            vecs = self.encoder.encode(self.names).astype(np.float32)
            np.save(path, vecs.astype(np.float16))
            print(f"[UMLSConceptIndex] done in {time.time() - t0:.1f}s")

        self._vecs = vecs
        self._index = FaissIndex(vecs.shape[1])
        self._index.build(vecs)

    # ---- 連結 -----------------------------------------------------
    def link(self, term: str, k: int = 5, sim_threshold: float = 0.0) -> List[Dict]:
        """回傳 [{cui, umls_name, score, match}]（score 由高到低）。"""
        term = (term or "").strip()
        if not term:
            return []

        norm = _normalize(term)
        if norm in self._exact:
            cui = self._exact[norm]
            return [{"cui": cui, "umls_name": self.names[self.cuis.index(cui)],
                     "score": 1.0, "match": "exact"}]

        self._ensure_index()
        qvec = self.encoder.encode([term])[0]
        ids, sims = self._index.search(qvec, k)
        hits: List[Dict] = []
        for idx, sim in zip(ids, sims):
            if idx < 0 or idx >= len(self.cuis):
                continue
            sim = float(sim)
            if sim < sim_threshold:
                continue
            hits.append({"cui": self.cuis[idx], "umls_name": self.names[idx],
                         "score": sim, "match": "semantic"})
        return hits

    def link_best(self, term: str, sim_threshold: float) -> Optional[Dict]:
        hits = self.link(term, k=5)
        if not hits:
            return None
        best = hits[0]
        if best["match"] == "exact" or best["score"] >= sim_threshold:
            return best
        return None
