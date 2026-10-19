# === config.py ===
import os


def _load_dotenv(path: str = None) -> None:
    """極簡 .env 載入器（無第三方相依）；只在 key 尚未存在於環境時才設定。"""
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                # 去掉行內註解： FOO=bar   # comment
                if " #" in v:
                    v = v.split(" #", 1)[0]
                elif "\t#" in v:
                    v = v.split("\t#", 1)[0]
                k, v = k.strip(), v.strip().strip('"').strip("'")
                os.environ.setdefault(k, v)
    except OSError:
        pass


_load_dotenv()

# === MIMIC-IV 根目錄 ===
MIMIC4_ROOT = os.getenv("MIMIC4_ROOT", "./data/mimic-iv-3.1/hosp")

# === 編碼映射檔 ===
CONDITION_MAPPING_FILE = os.getenv(
    "CONDITION_MAPPING_FILE", "./resources/CCSCM.csv"
)
PROCEDURE_MAPPING_FILE = os.getenv(
    "PROCEDURE_MAPPING_FILE", "./resources/CCSPROC.csv"
)
DRUG_MAPPING_FILE = os.getenv(
    "DRUG_MAPPING_FILE", "./resources/ATC.csv"
)

# === 本地 UMLS 資料夾 ===
UMLS_DATA_DIR = os.getenv(
    "UMLS_DATA_DIR", "./resources/umls_data"
)

# === 資料快取路徑 ===
DATA_BASE_PATH = os.getenv(
    "DATA_BASE_PATH", "./exp_data/ccscm_ccsproc"
)
SAMPLE_DATA_PATH = os.getenv(
    "SAMPLE_DATA_PATH", "./exp_data/ccscm_ccsproc/sample"
)

def sample_dataset_path(dataset: str, task: str) -> str:
    return os.path.join(
        DATA_BASE_PATH, f"sample_dataset_{dataset}_{task}.pkl"
    )
    
def full_dataset_path(dataset: str, task: str) -> str:
    return os.path.join(
        DATA_BASE_PATH, f"full_dataset_{dataset}_{task}.pkl"
    )
    
def sample_bigger_dataset_path(dataset: str, task: str) -> str:
    return os.path.join(
        SAMPLE_DATA_PATH, f"sample_dataset_{dataset}_{task}.pkl"
    )

# ------------------------------------------------------------------
# Fusion & Scoring 參數
# ------------------------------------------------------------------
K_UMLS       = int(os.getenv("K_UMLS", 5))
K_PUBMED     = int(os.getenv("K_PUBMED", 5))
# 逐 entity 做 UMLS concept linking 時，cosine 相似度需 ≥ 此門檻才算「已連結到 CUI」
UMLS_LINK_SIM_THRESHOLD = float(os.getenv("UMLS_LINK_SIM_THRESHOLD", 0.60))
DECAY_LAMBDA = float(os.getenv("DECAY_LAMBDA", 0.1))
ALPHA_SIM    = float(os.getenv("ALPHA_SIM", 0.7))
BETA_RECENCY = float(os.getenv("BETA_RECENCY", 0.3))
SUPPORT_RETRIEVAL_WEIGHT = float(os.getenv("SUPPORT_RETRIEVAL_WEIGHT", 0.35))
SUPPORT_NLI_WEIGHT       = float(os.getenv("SUPPORT_NLI_WEIGHT", 0.30))
SUPPORT_LITERATURE_WEIGHT = float(os.getenv("SUPPORT_LITERATURE_WEIGHT", 0.20))
SUPPORT_KB_WEIGHT        = float(os.getenv("SUPPORT_KB_WEIGHT", 0.15))
SUPPORT_SCORE_THRESHOLD  = float(os.getenv("SUPPORT_SCORE_THRESHOLD", 0.55))
SUPPORT_FILTER_ENABLED   = os.getenv("SUPPORT_FILTER_ENABLED", "1") == "1"
# 三向閾值：接受 / 審核 / 拒絕
SUPPORT_ACCEPT_THRESHOLD = float(os.getenv("SUPPORT_ACCEPT_THRESHOLD", 0.75))
SUPPORT_REJECT_THRESHOLD = float(os.getenv("SUPPORT_REJECT_THRESHOLD", 0.45))
SUPPORT_REVIEW_ENABLED   = os.getenv("SUPPORT_REVIEW_ENABLED", "1") == "1"

# ------------------------------------------------------------------
# Phase 2a — PubMed retrieval relevance gate
# ------------------------------------------------------------------
# 抓回的 abstract 與 patient context 的 cosine 相似度低於此值 → 直接丟棄，不送生成
PUBMED_MIN_SIM = float(os.getenv("PUBMED_MIN_SIM", 0.30))

# ------------------------------------------------------------------
# 多跳知識圖擴張（Patient → EHR entity → 文獻概念 → 再下一層…）
# ------------------------------------------------------------------
KG_MAX_HOPS      = int(os.getenv("KG_MAX_HOPS", 3))     # 從 EHR entity 往外擴幾層
KG_FRONTIER_TOPK = int(os.getenv("KG_FRONTIER_TOPK", 8))  # 每層最多挑幾個概念繼續往外擴
KG_HOP_K_PUBMED  = int(os.getenv("KG_HOP_K_PUBMED", 5))   # 每個 frontier 概念抓幾篇 PubMed

# ------------------------------------------------------------------
# Phase 2b — Triple ↔ sentence entailment verifier（本地 biomedical NLI）
# ------------------------------------------------------------------
VERIFIER_ENABLED          = os.getenv("VERIFIER_ENABLED", "1") == "1"
VERIFIER_MODEL            = os.getenv("VERIFIER_MODEL", "pritamdeka/PubMedBERT-MNLI-MedNLI")
# 找不到 / 載入失敗時的退回模型（通用 NLI）
VERIFIER_FALLBACK_MODEL   = os.getenv("VERIFIER_FALLBACK_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
# NLI 輸出很兩極（真 entailment ≈ 0.99），門檻設高不太傷 recall，卻能砍掉「只是提到主題」
# 的假陽性（SciFact dev 上：accept 0.6→0.9 讓 NOINFO recall 0.14→0.20，SUPPORT recall 幾乎不動）
VERIFIER_ACCEPT_THRESHOLD = float(os.getenv("VERIFIER_ACCEPT_THRESHOLD", 0.90))
VERIFIER_REVIEW_THRESHOLD = float(os.getenv("VERIFIER_REVIEW_THRESHOLD", 0.50))
# 支持句對 claim 的「矛盾」機率高於此 → 標為 CONTRADICTED（語意矛盾過濾）
VERIFIER_CONTRA_THRESHOLD = float(os.getenv("VERIFIER_CONTRA_THRESHOLD", 0.70))
VERIFIER_MAX_SENTENCES    = int(os.getenv("VERIFIER_MAX_SENTENCES", 25))
VERIFIER_BATCH_SIZE       = int(os.getenv("VERIFIER_BATCH_SIZE", 16))
VERIFIER_DEVICE           = os.getenv("VERIFIER_DEVICE", "")  # ""=自動；可填 "cpu" / "cuda:0"

# === PubMed / NCBI API Key ===
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")

# ------------------------------------------------------------------
# Gemini 設定（金鑰一律從環境變數 / .env 讀，不硬編碼）
# ------------------------------------------------------------------
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL     = os.getenv("GEMINI_MODEL", "gemma-3-4b-it")   # 論文結果所用模型
# 免費額度很小：每次 Gemini 呼叫之間至少間隔幾秒（把 RPM 壓在額度內）
GEMINI_MIN_INTERVAL_SEC = float(os.getenv("GEMINI_MIN_INTERVAL_SEC", 7.0))
GEMINI_MAX_RETRIES      = int(os.getenv("GEMINI_MAX_RETRIES", 4))
GEMINI_BATCH_SIZE       = int(os.getenv("GEMINI_BATCH_SIZE", 4))   # 每次呼叫塞幾篇 abstract
GEMINI_ABSTRACT_CHARS   = int(os.getenv("GEMINI_ABSTRACT_CHARS", 1500))  # 單篇 abstract 截斷長度

# ------------------------------------------------------------------
# Triple 生成後端切換
#   gemini        （預設，論文結果）→ Google Generative AI API，模型為 GEMINI_MODEL
#   openai_compat  → 任何提供 /v1/chat/completions 的服務
#                    （Ollama / vLLM / llama.cpp server / LM Studio）；
#                    optional alternative backend, not used for the reported results
# ------------------------------------------------------------------
TRIPLE_LLM_BACKEND = os.getenv("TRIPLE_LLM_BACKEND", "gemini").strip().lower()
LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL    = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b-instruct")
LOCAL_LLM_API_KEY  = os.getenv("LOCAL_LLM_API_KEY", "ollama")   # Ollama 不驗證，隨便填
LOCAL_LLM_TIMEOUT  = float(os.getenv("LOCAL_LLM_TIMEOUT", 180))
LOCAL_LLM_MIN_INTERVAL_SEC = float(os.getenv("LOCAL_LLM_MIN_INTERVAL_SEC", 0.0))

#-----------------------------
TASKS = {
    "drugrec": {
        "pkl": "exp_data/ccscm_ccsproc/sample_dataset_mimic4_drugrec_preprocessed.pkl",
        "loss": "bce",
    },
    # 以下三個由 build_labels.py 產生（任務定義見該檔）
    "mortality": {
        "pkl": "exp_data/labels/mortality.pkl",
        "loss": "bce",
    },
    "readmission": {
        "pkl": "exp_data/labels/readmission.pkl",
        "loss": "bce",
    },
    "lenofstay": {
        "pkl": "exp_data/labels/lenofstay.pkl",
        "loss": "ce",
    },
}


