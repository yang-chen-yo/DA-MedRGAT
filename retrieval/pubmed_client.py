# === retrieval/pubmed_client.py  (帶自動重試與指數退避) ===
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

__all__ = ["PubMedClient", "search_pubmed"]


class PubMedClient:
    """簡易封裝 NCBI E-utilities (PubMed)，內建自動重試 + 指數退避。"""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(
        self,
        email: str,
        api_key: Optional[str] = None,
        *,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
    ):
        """
        Args
        ----
        email          : 你的聯絡信箱（遵守 NCBI 規範）
        api_key        : NCBI 個人 API Key（可提高限額，選填）
        max_retries    : 連線或 5xx 失敗時，最高重試次數
        backoff_factor : 指數退避係數；1→1,2,4,8… 秒
        """
        if not email:
            raise ValueError("`email` 為必填，以符合 NCBI E-utilities 使用規範")
        self.email = email
        self.api_key = api_key

        # -------- 建立帶重試的 Session ----------
        retry_cfg = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        self.session = requests.Session()
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    # ------------------------------------------------------------------
    # ESearch：依關鍵字取得 PMID 清單
    # ------------------------------------------------------------------
    def search(
        self,
        term: str,
        *,
        retmax: int = 5,
        sort: str = "relevance",   # PubMed Best Match，比預設的「最新」更相關
        sleep_sec: float = 0.11,   # ≤3 req/s
    ) -> List[str]:
        # 1) 準備參數
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": retmax,
            "retmode": "json",
            "sort": sort,
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        # 2) 發 request（Adapter 會自動重試連線階段的 429/5xx；串流讀到一半
        #    斷掉的 ChunkedEncodingError 等 adapter 接不到，自己補重試，
        #    避免一次網路抖動炸掉跑好幾小時的整批任務）
        resp = None
        last_err: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/esearch.fcgi",
                    params=params,
                    timeout=20,
                )
                break
            except requests.exceptions.RequestException as e:
                last_err = e
                wait = min(2 ** attempt, 20)
                print(f"[WARN] PubMed esearch 失敗（第 {attempt}/3 次，term={term!r}）："
                      f"{e}；{wait}s 後重試")
                time.sleep(wait)
        if resp is None:
            print(f"[WARN] PubMed esearch 重試仍失敗，term={term!r}：{last_err}")
            return []

        # 3) 處理非 200 回應
        if resp.status_code != 200:
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                print(f"[WARN] PubMed 429 限流，等待 {wait}s")
                time.sleep(wait)
            else:
                print(f"[WARN] PubMed HTTP {resp.status_code} for term={term!r}")
            return []

        # 4) 安全解析 JSON
        try:
            data = resp.json()
        except ValueError:
            snippet = resp.text[:100].replace("\n", " ")
            print(f"[WARN] PubMed 回傳非 JSON（前 100 字）：{snippet!r}")
            return []

        # 5) 正常取 idlist 並限速
        pmids = data.get("esearchresult", {}).get("idlist", [])
        time.sleep(sleep_sec)
        return pmids

    # ------------------------------------------------------------------
    # EFetch：依 PMID 抓取 XML，解析標題與摘要
    # ------------------------------------------------------------------
    def fetch_abstracts(
        self,
        pmids: List[str],
        *,
        sleep_sec: float = 0.11,
        max_attempts: int = 5,
    ) -> List[Dict[str, str]]:
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "email": self.email,
        }
        if self.api_key:
            params["api_key"] = self.api_key

        # requests/urllib3 的 Retry adapter 只涵蓋「連線階段」的失敗；
        # 串流回應讀到一半斷掉（ChunkedEncodingError / ProtocolError）是在
        # 讀取 .text 時才炸，adapter 接不到，要自己包重試，否則一次網路
        # 抖動就會把跑幾小時的整批任務直接炸掉。
        xml_text = None
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = self.session.get(
                    f"{self.BASE_URL}/efetch.fcgi", params=params, timeout=30
                )
                resp.raise_for_status()
                xml_text = resp.text
                break
            except (requests.exceptions.RequestException, ET.ParseError) as e:
                last_err = e
                wait = min(2 ** attempt, 30)
                print(f"[WARN] PubMed efetch 失敗（第 {attempt}/{max_attempts} 次，"
                      f"{len(pmids)} 個 PMID）：{e}；{wait}s 後重試")
                time.sleep(wait)

        if xml_text is None:
            print(f"[WARN] PubMed efetch 重試 {max_attempts} 次仍失敗，"
                  f"跳過這批 {len(pmids)} 個 PMID：{last_err}")
            return []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            print(f"[WARN] PubMed efetch 回傳的 XML 解析失敗，跳過這批：{e}")
            return []
        articles: List[Dict[str, str]] = []
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID") or ""
            title = (art.findtext(".//ArticleTitle") or "").strip()

            # 摘要可能分多段
            abstract_parts = [seg.text or "" for seg in art.findall(".//AbstractText")]
            abstract = " ".join(p.strip() for p in abstract_parts if p.strip())

            # 先嘗試取 ArticleDate，若無再退回 PubDate
            ad = art.find(".//ArticleDate")
            if ad is not None:
                ay = ad.findtext("Year")
                am = ad.findtext("Month")
                ady = ad.findtext("Day")
            else:
                pub = art.find(".//PubDate")
                ay = pub.findtext("Year") or None
                am = pub.findtext("Month") or None
                ady = pub.findtext("Day") or None

            # 將英文月份縮寫轉成數字
            month_map = {
                'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06',
                'Jul':'07','Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12'
            }
            if ay and am and ady:
                m = month_map.get(am, am.zfill(2))
                d = ady.zfill(2)
                timestamp = f"{ay}-{m}-{d}"
            elif ay:
                timestamp = f"{ay}-01-01"
            else:
                timestamp = None

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": int(ay) if ay and ay.isdigit() else None,
                    "timestamp": timestamp,
                }
            )

        time.sleep(sleep_sec)
        return articles


# ----------------------------------------------------------------------
# Convenience wrapper：多關鍵字批次搜尋
# ----------------------------------------------------------------------
def search_pubmed(
    keywords: List[str],
    *,
    email: str,
    api_key: Optional[str] = None,
    retmax: int = 5,
) -> List[Dict[str, str]]:
    client = PubMedClient(email=email, api_key=api_key)
    out: List[Dict[str, str]] = []
    for kw in keywords:
        pmids = client.search(kw, retmax=retmax)
        for art in client.fetch_abstracts(pmids):
            art["keyword"] = kw
            out.append(art)
    return out

