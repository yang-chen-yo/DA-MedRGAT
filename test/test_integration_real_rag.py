import os
import sys
import unittest

# 把專案根目錄加入到模組搜尋路徑
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from fusion.triple_generation_hf import generate_triples_local_llama2

class TestRealRAGIntegration(unittest.TestCase):
    def test_real_llm_default_model(self):
        """
        Integration test: use the default RAG model to generate triples,
        and print out the actual output. If inference fails, the test fails.
        """
        patient_context = (
            "PatientID: P0001; Visit Date: 2025-07-09; "
            "Diagnoses: I10; Medications: 2025-07-10 Lisinopril (C09AA02); "
            "Labs: BloodPressure=160/100 mmHg."
        )
        abstracts = [
            "Lisinopril has been shown to effectively reduce systolic blood pressure "
            "in patients with hypertension (I10)."
        ]

        # 不再 skipTest，一旦出錯就 fail
        try:
            triples = generate_triples_local_llama2(
                patient_context=patient_context,
                abstracts=abstracts,
                umls_facts=None
            )
        except Exception as e:
            print(f"\n[ERROR] LLM 推理失敗：{e}\n")
            self.fail(f"LLM 推理失敗：{e}")

        # 印出真實推理結果
        print("\n=== Generated Triples ===")
        for t in triples:
            print(t)
        print("=========================\n")

        # 最基本的型態檢查
        self.assertIsInstance(triples, list)
        self.assertGreater(len(triples), 0, "Should generate at least one triple")
        first = triples[0]
        for key in ["head", "head_type", "relation", "tail", "tail_type", "source"]:
            self.assertIn(key, first)

if __name__ == '__main__':
    unittest.main()

