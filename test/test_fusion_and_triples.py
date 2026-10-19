import os
import sys
import tempfile
import pickle
import json
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# ========== 調整路徑 ==========
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

import numpy as np

from fusion.scoring import cosine_similarity, recency_weight, score_pubmed_hit
from fusion.evidence import assess_triple_support
from fusion.merger import fuse_and_score_progress, merge_to_triples
from fusion.triple_generation_hf import generate_triples_local_llama2
from kg.triple import (
    samples_to_triples,
    load_preprocessed_samples,
    load_triples_from_samples,
    Triple,
)
from data.triple_loader import load_triples
from config import sample_dataset_path


# -------------------------------------------------------------------
class TestFusionScoring(unittest.TestCase):
    def test_cosine_similarity(self):
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        sim = cosine_similarity(a, b)
        print("\n[cosine_similarity] =", sim)
        self.assertAlmostEqual(sim, 1.0)

    def test_recency_weight(self):
        current = datetime.now().year
        w = recency_weight(current)
        print("[recency_weight] =", w)
        self.assertAlmostEqual(w, 1.0)

    def test_score_pubmed_hit(self):
        current = datetime.now().year
        score = score_pubmed_hit(1.0, current)
        print("[score_pubmed_hit] =", score)
        self.assertAlmostEqual(score, 1.0)

    def test_assess_triple_support_uses_literature_evidence(self):
        triple = Triple(
            head="Patient:10000032",
            head_type="Patient",
            relation="HAS_DISEASE",
            tail="Disease:Type 2 Diabetes",
            tail_type="Disease",
        )
        result = assess_triple_support(
            triple=triple,
            retrieval_score=0.92,
            patient_context="patient with type 2 diabetes and poor glycemic control",
            evidence_text="Type 2 diabetes is associated with poor glycemic control and increased cardiovascular risk.",
            evidence_title="Type 2 Diabetes and Glycemic Control",
            evidence_source="PubMed",
        )
        self.assertIn("literature", result.components)
        self.assertGreater(result.components["literature"], 0.4)
        self.assertGreater(result.score, 0.65)


# -------------------------------------------------------------------
class TestKGTriples(unittest.TestCase):
    def test_samples_to_triples(self):
        samples = [
            {
                'patient_id': 'p1',
                'conditions': ['C1'],
                'procedures': ['P1'],
                'drugs': ['D1'],
                'visit_time': '2025-07-08'
            }
        ]
        triples = samples_to_triples(samples)
        print("\n[samples_to_triples]")
        for t in triples:
            print(" ", t)
        self.assertEqual(len(triples), 3)
        self.assertEqual(triples[0].head, 'Patient:p1')
        self.assertEqual(triples[0].relation, 'HAS_DISEASE')

    def test_load_triples_from_samples(self):
        # 建立暫存 pickle
        samples = [
            {'patient_id': 'p2', 'conditions': ['C2'], 'procedures': [], 'drugs': [], 'visit_time': None}
        ]
        tmp = tempfile.NamedTemporaryFile(delete=False)
        pickle.dump(samples, tmp)
        tmp.close()

        with patch('config.sample_dataset_path', return_value=tmp.name):
            triples = load_triples_from_samples('x', 'y')
        os.unlink(tmp.name)

        print("\n[load_triples_from_samples]")
        for t in triples:
            print(" ", t)
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0].tail, 'Disease:C2')

    def test_load_triples_preserves_provenance_fields(self):
        payload = [{
            'head': 'Patient:10000032',
            'head_type': 'Patient',
            'relation': 'HAS_DISEASE',
            'tail': 'Disease:Example',
            'tail_type': 'Disease',
            'visit_id': '22595853',
            'visit_date': '2180-05-06',
            'timestamp': '2025-07-10',
            'source': 'PubMed',
            'evidence_source': 'PubMed',
            'evidence_id': '40640734',
            'evidence_rank': 1,
            'evidence_score': 0.92,
            'evidence_title': 'Example Title',
            'evidence_text': 'Example abstract'
        }]
        with tempfile.TemporaryDirectory() as tmpdir:
            patient_dir = os.path.join(tmpdir, '10000032')
            os.makedirs(patient_dir, exist_ok=True)
            with open(os.path.join(patient_dir, 'triples_22595853.json'), 'w', encoding='utf-8') as f:
                json.dump(payload, f)

            triples = load_triples(tmpdir)

        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0]['evidence_id'], '40640734')
        self.assertEqual(triples[0]['evidence_source'], 'PubMed')


# -------------------------------------------------------------------
class TestTripleGenerationHF(unittest.TestCase):
    @patch('fusion.triple_generation_hf.pipeline')
    @patch('fusion.triple_generation_hf.AutoModelForCausalLM')
    @patch('fusion.triple_generation_hf.AutoTokenizer')
    def test_generate_triples_local_llama2(self, mock_tok, mock_model, mock_pipe):
        fake_out = (
            'xxx[{{"head":"Patient:p","head_type":"Patient","relation":"R",'
            '"tail":"T","tail_type":"Type","timestamp":"2025-07-01","source":"X"}}]'
        )
        fake_gen = MagicMock(return_value=[{'generated_text': fake_out}])
        mock_pipe.return_value = fake_gen

        triples = generate_triples_local_llama2('ctx', ['abs'], ['fact'])
        print("\n[generate_triples_local_llama2]", triples)

        self.assertIsInstance(triples, list)
        self.assertEqual(triples[0]['relation'], 'R')


# -------------------------------------------------------------------
class TestMerger(unittest.TestCase):
    # merge_to_triples 會寫 retrieval_cache/ 與 triples_output/（相對路徑），
    # 在暫存目錄裡跑，避免寫進真正的資料，也避免上次的快取讓測試跳過
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    @patch('fusion.merger.PUBMED_MIN_SIM', -1.0)   # 測試裡不啟用相關性 gate
    @patch('fusion.merger.get_verifier', return_value=None)
    @patch('fusion.merger.assess_triple_support')
    @patch('fusion.merger.UMLSConceptIndex')
    @patch('fusion.merger.PubMedClient')
    @patch('fusion.merger._rag_generate')
    def test_fuse_and_merge(self, mock_gen, mock_pub, mock_index, mock_support, mock_verifier):
        from fusion.evidence import SupportResult
        mock_support.return_value = SupportResult(score=0.9, status='ACCEPTED', components={})
        # 偽造逐 entity 的 UMLS linking
        idx = mock_index.shared.return_value
        idx.link_best.return_value = {
            'cui': 'C0011849', 'umls_name': 'diabetes mellitus',
            'score': 0.91, 'match': 'semantic',
        }

        # 偽造 PubMedClient
        p = mock_pub.return_value
        p.search.return_value = ['123']
        p.fetch_abstracts.return_value = [{
            'pmid': '123', 'title': 'T', 'abstract': 'A',
            'year': datetime.now().year,
        }]

        # RAG（Gemini）生成結果：Type 2 Diabetes 是 anchor，polyuria 是文獻新概念
        mock_gen.return_value = [{
            'head': 'Type 2 Diabetes', 'head_type': 'Disease', 'relation': 'HAS_SYMPTOM',
            'tail': 'polyuria', 'tail_type': 'Symptom', 'source': 'RAG',
            '_abstract_id': '123',
        }]

        patient_fields = {
            'conditions': [[{'code': 'E11', 'name': 'Type 2 Diabetes'}]],
            'procedures': [], 'drugs': [],
        }
        fused = fuse_and_score_progress('ctx', 'dir', 'email', patient_fields)
        triples = merge_to_triples('p1', visit_id='default_visit', fused=fused,
                                   patient_context='ctx', patient_fields=patient_fields,
                                   accumulate=False)

        print("\n[fuse_and_score] =>", fused)
        print("[merge_to_triples]")
        for t in triples:
            print(" ", t)

        self.assertEqual(len(triples), 2)
        self.assertEqual(triples[0].relation, 'HAS_DISEASE')
        self.assertEqual(triples[0].tail, 'Type 2 Diabetes')
        self.assertEqual(triples[0].evidence_id, 'C0011849')
        self.assertEqual(triples[0].source, 'EHR')
        # RAG triple 經正規化 + 病人錨定
        self.assertEqual(triples[1].relation, 'HAS_SYMPTOM')
        self.assertEqual(triples[1].head, 'Type 2 Diabetes')
        self.assertEqual(triples[1].tail, 'polyuria')
        self.assertEqual(triples[1].source, 'RAG')


# -------------------------------------------------------------------
if __name__ == '__main__':
    unittest.main()

