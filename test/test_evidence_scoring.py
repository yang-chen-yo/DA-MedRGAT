import os
import sys
import unittest
from unittest.mock import patch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_dir)

from fusion.triple_verifier import (
    verbalize_triple,
    split_sentences,
    TripleEvidenceVerifier,
    _find_entail_index,
    _find_label_index,
)
from kg.triple import Triple


def _cols(entail_list, contras=None):
    """把 entailment 機率列表轉成 3 類 softmax rows（contra, entail, neutral）。"""
    contra_list = contras or [0.0] * len(entail_list)
    rows = []
    for e, c in zip(entail_list, contra_list):
        n = max(0.0, 1.0 - e - c)
        rows.append([c, e, n])  # idx0=contra, idx1=entail, idx2=neutral
    return rows


class TestVerbalize(unittest.TestCase):
    def test_relation_templates(self):
        self.assertEqual(
            verbalize_triple({'head': 'Metformin', 'relation': 'TREATS', 'tail': 'type 2 diabetes'}),
            'Metformin is used to treat type 2 diabetes.',
        )
        self.assertEqual(
            verbalize_triple({'head': 'P', 'relation': 'Patient–Disease', 'tail': 'hepatitis'}),
            'The patient has hepatitis.',
        )
        self.assertEqual(
            verbalize_triple({'head': 'Warfarin', 'relation': 'Drug–AdverseEffect', 'tail': 'bleeding'}),
            'Warfarin can cause bleeding as an adverse effect.',
        )

    def test_type_prefix_stripped(self):
        out = verbalize_triple({'head': 'Drug:Metformin', 'relation': 'TREATS', 'tail': 'Disease:diabetes'})
        self.assertEqual(out, 'Metformin is used to treat diabetes.')

    def test_unknown_relation_fallback(self):
        out = verbalize_triple({'head': 'x', 'relation': 'FOO_BAR', 'tail': 'y'})
        self.assertEqual(out, 'x foo bar y.')

    def test_accepts_triple_dataclass(self):
        t = Triple(head='Aspirin', head_type='Drug', relation='CAUSES_SIDE_EFFECT',
                   tail='gastric ulcer', tail_type='AdverseEffect')
        self.assertEqual(verbalize_triple(t), 'Aspirin can cause gastric ulcer as an adverse effect.')


class TestSentenceSplit(unittest.TestCase):
    def test_basic(self):
        s = split_sentences('Metformin reduces HbA1c. It is well tolerated. No severe events.')
        self.assertEqual(len(s), 3)

    def test_abbreviation_not_split(self):
        s = split_sentences('It is well tolerated (e.g. mild GI upset). No severe events.')
        self.assertEqual(len(s), 2)

    def test_max_sentences(self):
        s = split_sentences('A. B. C. D. E.', max_sentences=2)
        self.assertEqual(len(s), 2)

    def test_empty(self):
        self.assertEqual(split_sentences(''), [])
        self.assertEqual(split_sentences(None), [])


class TestEntailIndex(unittest.TestCase):
    def test_finds_entailment_label(self):
        class Cfg:
            id2label = {0: 'CONTRADICTION', 1: 'NEUTRAL', 2: 'ENTAILMENT'}
            num_labels = 3
        self.assertEqual(_find_entail_index(Cfg()), 2)
        self.assertEqual(_find_label_index(Cfg(), 'contra'), 0)

    def test_deberta_order(self):
        class Cfg:
            id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            num_labels = 3
        self.assertEqual(_find_entail_index(Cfg()), 0)
        self.assertEqual(_find_label_index(Cfg(), 'contra'), 2)

    def test_pubmedbert_mednli_order(self):
        class Cfg:
            id2label = {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
            num_labels = 3
        self.assertEqual(_find_entail_index(Cfg()), 1)
        self.assertEqual(_find_label_index(Cfg(), 'contra'), 0)


class TestVerifyDecision(unittest.TestCase):
    """不載入真模型，直接注入 entailment 機率測 decision 分流。"""

    def _verifier(self):
        v = TripleEvidenceVerifier(accept_threshold=0.60, review_threshold=0.40,
                                   contra_threshold=0.55)
        v._loaded_name = 'mock-nli'
        v._entail_idx, v._contra_idx = 1, 0
        return v

    def _run(self, probs, tail='diabetes', contras=None):
        v = self._verifier()
        # 每一句都提到 tail，讓「實體接地」通過，單純測 NLI 分數 → decision
        with patch.object(TripleEvidenceVerifier, '_softmax_cols',
                          return_value=_cols(probs, contras)):
            return v.verify(
                {'head': 'Metformin', 'relation': 'TREATS', 'tail': tail},
                evidence_text=('Metformin is studied in diabetes. '
                               'Metformin lowers glucose in diabetes. '
                               'It is first-line for type 2 diabetes.'),
                evidence_title='Metformin and diabetes',
                evidence_id='123',
            )

    def test_accepted(self):
        r = self._run([0.1, 0.2, 0.2, 0.92])
        self.assertEqual(r.decision, 'ACCEPTED')
        self.assertAlmostEqual(r.score, 0.92, places=4)
        self.assertIn('first-line', r.supporting_sentence)

    def test_review(self):
        r = self._run([0.1, 0.1, 0.48, 0.2])
        self.assertEqual(r.decision, 'REVIEW')

    def test_rejected(self):
        r = self._run([0.05, 0.1, 0.1, 0.2])
        self.assertEqual(r.decision, 'REJECTED')

    def test_high_nli_but_tail_not_grounded_is_review(self):
        v = self._verifier()
        with patch.object(TripleEvidenceVerifier, '_softmax_cols',
                          return_value=_cols([0.1, 0.97])):
            r = v.verify(
                {'head': 'Spironolactone', 'relation': 'TREATS', 'tail': 'systemic inflammation'},
                evidence_text='Spironolactone reduces myocardial fibrosis in HFpEF.',
                evidence_title='MRA trial',
            )
        self.assertEqual(r.decision, 'REVIEW')  # NLI 高但 tail 沒出現在句子 → 降級

    def test_contradiction_detected(self):
        v = self._verifier()
        # 句子提到 diabetes（tail 接地），但對 claim 是「矛盾」
        with patch.object(TripleEvidenceVerifier, '_softmax_cols',
                          return_value=_cols([0.05, 0.05, 0.1, 0.1],
                                             contras=[0.1, 0.1, 0.85, 0.1])):
            r = v.verify(
                {'head': 'Metformin', 'relation': 'TREATS', 'tail': 'diabetes'},
                evidence_text=('Metformin is studied in diabetes. '
                               'Metformin lowers glucose in diabetes. '
                               'Metformin does not improve outcomes in diabetes. '
                               'More trials in diabetes are needed.'),
                evidence_title='Metformin and diabetes',
            )
        self.assertEqual(r.decision, 'CONTRADICTED')

    def test_no_evidence_rejected(self):
        v = self._verifier()
        r = v.verify({'head': 'a', 'relation': 'r', 'tail': 'b'}, evidence_text='', evidence_title=None)
        self.assertEqual(r.decision, 'REJECTED')
        self.assertEqual(r.score, 0.0)


if __name__ == '__main__':
    unittest.main()
