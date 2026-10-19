#fusion/scoring.py
import numpy as np
from datetime import datetime
from config import DECAY_LAMBDA, ALPHA_SIM, BETA_RECENCY


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    計算兩向量的餘弦相似度。要求 a, b 已 normalize。
    """
    return float(np.dot(a, b))


def recency_weight(pub_year: int, current_year: int = None) -> float:
    """
    根據發表年份計算衰減權重：
        w = exp(-DECAY_LAMBDA * Δyear)
    DECAY_LAMBDA 由 config 設定。
    """
    if current_year is None:
        current_year = datetime.now().year
    delta = current_year - pub_year
    return float(np.exp(-DECAY_LAMBDA * delta))


def score_pubmed_hit(sim: float, pub_year: int) -> float:
    """
    結合相似度與發表年份權重給出最終分數：
        score = ALPHA_SIM * sim + BETA_RECENCY * recency_weight
    ALPHA_SIM, BETA_RECENCY 由 config 設定。
    """
    rw = recency_weight(pub_year)
    return float(ALPHA_SIM * sim + BETA_RECENCY * rw)
