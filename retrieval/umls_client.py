import os
import pickle
import csv
from typing import List, Dict
from config import UMLS_DATA_DIR

# Lazy import Spark NLP and pyspark to avoid import errors
try:
    import sparknlp
    from pyspark.sql import SparkSession
    from sparknlp.pretrained import PretrainedPipeline
except ImportError:
    sparknlp = None
    SparkSession = None
    PretrainedPipeline = None


class UMLSClient:
    """
    Loader for UMLS static concepts and optional dynamic ICD10CM→CUI mapping via Spark NLP.
    同時支援載入 UMLS 關係三元組 (relation triples CSV)。

    靜態文件:
      - concepts.txt
      - concept_names.txt
      - relations.txt
      - umls.csv (關係三元組: relation\tCUI1\tCUI2\tweight)
    動態 code→CUI:
      - 使用 Spark NLP icd10cm_umls_mapping pipeline 或注入自訂 pipeline
    """

    def __init__(
        self,
        umls_dir: str = UMLS_DATA_DIR,
        cache_dir: str = None,
        model: str = "icd10cm_umls_mapping",
        pipeline: PretrainedPipeline = None
    ):
        self.umls_dir = umls_dir
        self.cache_dir = cache_dir or umls_dir

        # 如果使用者注入 pipeline，直接使用
        if pipeline is not None:
            self.pipeline = pipeline
            self.spark = None
        else:
            # 檢查是否可用 Spark NLP
            if sparknlp and SparkSession and PretrainedPipeline:
                try:
                    spark = SparkSession.builder.getOrCreate()
                    _ = spark.sparkContext._jsc
                except Exception:
                    try:
                        spark = sparknlp.start()
                    except Exception:
                        spark = None
                if spark:
                    self.spark = spark
                    try:
                        self.pipeline = PretrainedPipeline(model, lang="en", remote_loc=None)
                    except Exception:
                        self.pipeline = None
                else:
                    self.spark = None
                    self.pipeline = None
            else:
                self.spark = None
                self.pipeline = None

        # 載入靜態 UMLS 資料
        self.concepts = self._load_concepts()
        self.concept_names = self._load_concept_names()
        self.relation_types = self._load_relation_types()
        self.relation_triples = self._load_relation_triples()

    def _load_concepts(self) -> List[str]:
        cache = os.path.join(self.cache_dir, 'concepts.pkl')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                return pickle.load(f)
        path = os.path.join(self.umls_dir, 'concepts.txt')
        cuis = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                cui = line.strip()
                if cui:
                    cuis.append(cui)
        with open(cache, 'wb') as f:
            pickle.dump(cuis, f)
        return cuis

    def _load_concept_names(self) -> Dict[str, str]:
        cache = os.path.join(self.cache_dir, 'concept_names.pkl')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                return pickle.load(f)
        path = os.path.join(self.umls_dir, 'concept_names.txt')
        mapping: Dict[str, str] = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    cui, name = parts
                    mapping[cui] = name.lower()
        with open(cache, 'wb') as f:
            pickle.dump(mapping, f)
        return mapping

    def _load_relation_types(self) -> List[str]:
        cache = os.path.join(self.cache_dir, 'relation_types.pkl')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                return pickle.load(f)
        path = os.path.join(self.umls_dir, 'relations.txt')
        types: List[str] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                name = line.strip()
                if name:
                    types.append(name)
        with open(cache, 'wb') as f:
            pickle.dump(types, f)
        return types

    def _load_relation_triples(self) -> List[Dict[str, object]]:
        cache = os.path.join(self.cache_dir, 'relation_triples.pkl')
        if os.path.exists(cache):
            with open(cache, 'rb') as f:
                return pickle.load(f)
        path = os.path.join(self.umls_dir, 'umls.csv')
        triples: List[Dict[str, object]] = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 4:
                    rel, c1, c2, wt = row[0], row[1], row[2], row[3]
                    try:
                        weight = float(wt)
                    except ValueError:
                        weight = 1.0
                    triples.append({
                        'relation': rel,
                        'cui1': c1,
                        'cui2': c2,
                        'weight': weight
                    })
        with open(cache, 'wb') as f:
            pickle.dump(triples, f)
        return triples

    def query_by_codes(self, codes: List[str]) -> List[Dict[str, str]]:
        """
        給定 ICD-10CM codes list，使用 Spark NLP mapping pipeline 轉成 UMLS CUIs，並回傳概念名稱。
        若 pipeline 為 None，拋出 ImportError。
        """
        if not self.pipeline:
            raise ImportError("Spark NLP Pipeline 未初始化，無法執行 code→CUI 映射。")
        results: List[Dict[str, str]] = []
        doc = self.pipeline.annotate(" ".join(codes))
        icd_list = doc.get('icd10cm', [])
        umls_list = doc.get('umls', [])
        for code, cui in zip(icd_list, umls_list):
            name = self.concept_names.get(cui)
            if name:
                results.append({'code': code, 'cui': cui, 'name': name})
        return results

    def query_relations(self, cui: str) -> List[Dict[str, object]]:
        return [t for t in self.relation_triples if t['cui1'] == cui or t['cui2'] == cui]


def query_umls(codes, umls_dir=None, model=None):
    """
    Wrapper: 使用 UMLSClient 查詢 ICD-10CM codes 對應的 UMLS CUIs.
    """
    umls_dir = umls_dir or UMLS_DATA_DIR
    client = UMLSClient(umls_dir=umls_dir)
    return client.query_by_codes(codes)

