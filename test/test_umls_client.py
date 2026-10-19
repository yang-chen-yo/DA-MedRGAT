import os, sys, tempfile, shutil, unittest

# 調整路徑以載入專案模組
test_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(test_dir, '..'))
sys.path.insert(0, root_dir)

from retrieval.umls_client import UMLSClient

class TestUMLSClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 建立臨時 UMLS 資料夾並寫入基本測試檔案
        cls.tmp_dir = tempfile.mkdtemp(prefix='umls_test_')
        # concept.txt
        with open(os.path.join(cls.tmp_dir, 'concepts.txt'), 'w', encoding='utf-8') as f:
            f.write("C0000039\nC0000052\n")
        # concept_name.txt
        with open(os.path.join(cls.tmp_dir, 'concept_names.txt'), 'w', encoding='utf-8') as f:
            f.write("C0000039\t1,2-dipalmitoylphosphatidylcholine\n")
        # relation.txt
        with open(os.path.join(cls.tmp_dir, 'relations.txt'), 'w', encoding='utf-8') as f:
            f.write("may_cause\nbelongs_to_the_category_of\n")
        # umls.csv (relation triples)
        with open(os.path.join(cls.tmp_dir, 'umls.csv'), 'w', encoding='utf-8') as f:
            f.write("may_cause\tC0000039\tC0000052\t1.0\n")
        # 初始化 client (pipeline 可能會停用)
        cls.client = UMLSClient(umls_dir=cls.tmp_dir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir)

    def test_concepts_loaded(self):
        self.assertIn('C0000039', self.client.concepts)
        self.assertIn('C0000052', self.client.concepts)

    def test_concept_names(self):
        name = self.client.concept_names.get('C0000039')
        self.assertEqual(name, '1,2-dipalmitoylphosphatidylcholine')

    def test_relation_types(self):
        self.assertIn('may_cause', self.client.relation_types)
        self.assertIn('belongs_to_the_category_of', self.client.relation_types)

    def test_relation_triples(self):
        triples = self.client.relation_triples
        self.assertTrue(any(
            t['relation']=='may_cause' and t['cui1']=='C0000039' and t['cui2']=='C0000052'
            for t in triples
        ))

    def test_query_by_codes_fallback(self):
        with self.assertRaises(ImportError):
            self.client.query_by_codes(['M8950'])

    def test_query_by_codes_success_with_fake_pipeline(self):
        # 使用 FakePipeline 模擬映射 icd10cm→umls
        class FakePipeline:
            def annotate(self, text: str):
                codes = text.split()
                return {'icd10cm': codes, 'umls': ['C0000039' for _ in codes]}
        client = UMLSClient(umls_dir=self.tmp_dir, pipeline=FakePipeline())
        result = client.query_by_codes(['X123', 'Y456'])
        expected = [
            {'code': 'X123', 'cui': 'C0000039', 'name': '1,2-dipalmitoylphosphatidylcholine'},
            {'code': 'Y456', 'cui': 'C0000039', 'name': '1,2-dipalmitoylphosphatidylcholine'}
        ]
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()

