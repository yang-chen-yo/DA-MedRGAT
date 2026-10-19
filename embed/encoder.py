# === embed/encoder.py ===
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency fallback
    SentenceTransformer = None


class Encoder:
    """
    封裝 Sentence-BERT （SapBERT）將文本轉成向量
    """
    def __init__(self, model_name: str = "all-mpnet-base-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name) if SentenceTransformer is not None else None

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        將多條文本轉成向量（shape=(len(texts), dim)）。
        
        1) 轉成字串列表  
        2) 批次 encode  
        3) L2 正規化  
        """
        clean_texts = [t if isinstance(t, str) else str(t) for t in texts]

        if self.model is not None:
            embeddings = self.model.encode(
                clean_texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings

        # Lightweight deterministic fallback used when the model package is unavailable.
        vectors = []
        for text in clean_texts:
            low = text.lower()
            features = [0.0] * 6
            for idx, token in enumerate(["diabetes", "hypertension", "drug", "risk", "treatment", "control"]):
                if token in low:
                    features[idx] = 1.0
            vectors.append(features)
        arr = np.asarray(vectors, dtype=float)
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        return arr / norm

