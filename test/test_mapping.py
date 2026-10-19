# test_mapping.py

import os
import sys

# 1) 把專案根目錄加入模組搜尋路徑（假設 test/ 與 data/ 同層）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# 2) 匯入 ICD-10 函式庫
try:
    from simple_icd_10_cm import get_description
except ModuleNotFoundError:
    from simple_icd_10 import get_description

# 3) 匯入你原本的 mapping 與 enrich 函式
from data.loader import load_mappings, enrich_sample_with_names

def main():
    # 讀取 CSV mapping
    cond_map, proc_map, drug_map = load_mappings()

    # 測試用 sample
    sample = {
        "conditions": [["I10", "A00", "ZZZ"]],
        "procedures": [["0JH60", "ZZZ"]],
        "drugs":      [["C10AA01", "XXX"]],
    }

    # 執行 enrich
    enriched = enrich_sample_with_names(sample, cond_map, proc_map, drug_map)

    # 輸出結果
    print("field       | code      | name                             | 來源")
    print("------------|-----------|----------------------------------|----------------")
    for field in ("conditions", "procedures", "drugs"):
        for row in enriched[field]:
            for item in row:
                code = item["code"]
                name = item["name"]
                # 判斷來源：CSV? 套件? or 原碼保留
                if field == "conditions" and code in cond_map:
                    src = "CSV"
                elif field == "procedures" and code in proc_map:
                    src = "CSV"
                elif field == "drugs" and code in drug_map:
                    src = "CSV"
                elif name != code:
                    src = "ICD-10 fallback"
                else:
                    src = "原碼保留"
                print(f"{field:12}| {code:9}| {name:32}| {src}")

if __name__ == "__main__":
    main()

