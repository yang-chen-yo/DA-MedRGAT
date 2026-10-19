import os
import sys
import unittest
from requests.exceptions import RequestException, JSONDecodeError
from urllib3.exceptions import NewConnectionError, MaxRetryError

# 設定專案根目錄
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from retrieval.pubmed_client import search_pubmed

class TestPubMedClient(unittest.TestCase):
    def setUp(self):
        # NCBI E-utilities 要求提供聯絡 email，從環境變數讀取
        self.email = os.getenv('NCBI_EMAIL', '')

    def test_real_query_dpcc(self):
        """真實呼叫 PubMed API，查詢並印出第一篇文章摘要。"""
        try:
            results = search_pubmed(
                keywords=['2-dipalmitoylphosphatidylcholine'],
                email=self.email
            )
        except (RequestException, NewConnectionError, MaxRetryError, JSONDecodeError) as e:
            self.skipTest(f"網路或 API 失敗，跳過整合測試：{e}")
            return

        # 驗證至少取得 1 筆文章
        self.assertGreaterEqual(len(results), 1)
        art = results[0]

        # 印出結果
        print("\n=== PubMed 實際查詢結果 ===")
        print(f"PMID:    {art['pmid']}")
        print(f"Title:   {art['title']}")
        print(f"Abstract: {art['abstract'][:300]}...")  # 只印前 300 字

        # 確保摘要不是空的
        self.assertTrue(art['abstract'].strip(), "摘要不應為空")

if __name__ == '__main__':
    unittest.main()

