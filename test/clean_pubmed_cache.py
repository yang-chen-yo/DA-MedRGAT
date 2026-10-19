#!/usr/bin/env python3
"""
clean_pubmed_cache.py:
1) 刪除 retrieval_cache/<patient_id>/<pmid>/pubmed.json 內含 0 或 1 筆摘要的檔案
2) 刪除 retrieval_cache/<patient_id>/<pmid>/last_batch.txt，
   以及 retrieval_cache/ 或 retrieval_cache/<patient_id>/ 下的 last_batch.txt，若內容為 '0'

使用方式：
  chmod +x clean_pubmed_cache.py
  python clean_pubmed_cache.py --cache-root /path/to/retrieval_cache
"""
import os
import json
import argparse

def clean_cache(cache_root: str):
    stale_deleted = 0
    lastbatch_deleted = 0

    # 1) root-level last_batch.txt
    root_last = os.path.join(cache_root, 'last_batch.txt')
    if os.path.isfile(root_last):
        with open(root_last, 'r', encoding='utf-8') as f:
            content = f.read().lstrip('\ufeff').strip()
        print(f"root last_batch.txt 內容: '{content}'")
        if content == '1':
            os.remove(root_last)
            lastbatch_deleted += 1
            print(f"刪除: {root_last}")

    # 2) patient-level
    for patient_id in os.listdir(cache_root):
        patient_dir = os.path.join(cache_root, patient_id)
        if not os.path.isdir(patient_dir):
            continue

        # patient-level last_batch.txt
        p_last = os.path.join(patient_dir, 'last_batch.txt')
        if os.path.isfile(p_last):
            with open(p_last, 'r', encoding='utf-8') as f:
                content = f.read().lstrip('\ufeff').strip()
            print(f"{patient_id}/last_batch.txt 內容: '{content}'")
            if content == '0':
                os.remove(p_last)
                lastbatch_deleted += 1
                print(f"刪除: {p_last}")

        # 3) pmid-level: 刪除 pubmed.json 及 last_batch.txt
        for pmid in os.listdir(patient_dir):
            pmid_dir = os.path.join(patient_dir, pmid)
            if not os.path.isdir(pmid_dir):
                continue

            # pmid-level last_batch.txt
            pmid_last = os.path.join(pmid_dir, 'last_batch.txt')
            if os.path.isfile(pmid_last):
                with open(pmid_last, 'r', encoding='utf-8') as f:
                    content = f.read().lstrip('\ufeff').strip()
                print(f"{patient_id}/{pmid}/last_batch.txt 內容: '{content}'")
                if content == '0':
                    os.remove(pmid_last)
                    lastbatch_deleted += 1
                    print(f"刪除: {pmid_last}")

            # pmid-level pubmed.json
            json_path = os.path.join(pmid_dir, 'pubmed.json')
            if os.path.isfile(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    print(f"無法解析 JSON: {json_path}, 刪除中...")
                    os.remove(json_path)
                    stale_deleted += 1
                    continue
                count = len(data) if isinstance(data, list) else '非 list'
                print(f"{patient_id}/{pmid}/pubmed.json 筆數: {count}")
                if isinstance(data, list) and count <= 3:
                    os.remove(json_path)
                    stale_deleted += 1
                    print(f"刪除: {json_path}")

    print(f"清理完成：刪除 pubmed.json {stale_deleted} 個，刪除 last_batch.txt {lastbatch_deleted} 個")


def main():
    parser = argparse.ArgumentParser(description='清理 pubmed.json 與 last_batch.txt 快取')
    parser.add_argument('--cache-root', '-r', required=True, help='retrieval_cache 根目錄')
    args = parser.parse_args()
    clean_cache(args.cache_root)


if __name__ == '__main__':
    main()

