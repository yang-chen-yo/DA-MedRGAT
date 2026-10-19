# === embed/faiss_index.py ===
import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    faiss = None


typedef = np.ndarray


class _FallbackIPIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None

    def add(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, q: np.ndarray, k: int):
        if self.vectors is None:
            return np.zeros((1, k), dtype=np.int64), np.zeros((1, k), dtype=np.float32)
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        sims = self.vectors @ q.T
        sims = sims[:, 0]
        topk = np.argsort(sims)[::-1][:k]
        scores = sims[topk]
        return topk.reshape(1, -1), scores.reshape(1, -1)


class FaissIndex:
    """
    建立並查詢 FAISS 向量索引
    """
    def __init__(self, dim: int):
        if faiss is not None:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = _FallbackIPIndex(dim)

    def build(self, vectors: np.ndarray):
        """
        vectors: numpy array shape [N, dim]
        """
        self.index.add(vectors)

    def search(self, query_vec: np.ndarray, k: int = 5):
        """
        query_vec: shape [dim] 或 [1, dim]
        回傳 (indices, scores)
        """
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if faiss is not None:
            D, I = self.index.search(q, k)
            return I[0].tolist(), D[0].tolist()
        I, D = self.index.search(q, k)
        return I[0].tolist(), D[0].tolist()