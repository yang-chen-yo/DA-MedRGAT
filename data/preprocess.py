import numpy as np
import torch
from pyhealth.tokenizer import Tokenizer
from tqdm import tqdm

def flatten(nested_list):
    """
    Flatten a nested list of arbitrary depth to a flat list.
    """
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def multihot(label_indices, num_labels):
    """
    Convert a list of label indices to a multi-hot numpy vector.
    """
    vec = np.zeros(num_labels, dtype=int)
    for idx in label_indices:
        vec[idx] = 1
    return vec

def prepare_label(sample_dataset, drugs):
    """
    Prepare multi-hot label for drug recommendation.
    """
    tokenizer = Tokenizer(sample_dataset.get_all_tokens(key='drugs'))
    indices = tokenizer.convert_tokens_to_indices(drugs)
    num_labels = tokenizer.get_vocabulary_size()
    return multihot(indices, num_labels)

def prepare_drug_indices(sample_dataset):
    """
    Annotate each patient with 'drugs_ind' tensor (multi-hot).
    """
    for patient in tqdm(sample_dataset, desc="Preparing drug indices"):
        label = prepare_label(sample_dataset, patient['drugs'])
        patient['drugs_ind'] = torch.tensor(label, dtype=torch.float)
    return sample_dataset

def pad_and_convert(visits, max_visits, max_nodes):
    """
    Given a list of node-code lists (one per visit),
    produce a tensor of shape [max_visits, max_nodes] with multi-hot encoding,
    reversed in time (most recent first) and padded with zeros.
    """
    padded = []
    for codes in reversed(visits):
        vec = torch.zeros(max_nodes, dtype=torch.float)
        for code in codes:
            vec[code] = 1
        padded.append(vec)
    while len(padded) < max_visits:
        padded.append(torch.zeros(max_nodes, dtype=torch.float))
    return torch.stack(padded, dim=0)


def _as_visit_lists(val):
    """
    正規化成 List[List[str]]（每個內層 list = 一次就診的代碼清單）。

    task_fn.py 的累積邏輯（mortality / readmission / lenofstay）已經把
    conditions/procedures/drugs 疊成 [[visit0代碼...], [visit1代碼...], ...]，
    是「已經雙層」的結構；drug_recommendation_mimic4_fn 的 conditions_all /
    procedures_all / drugs_all 也是同樣的雙層結構。只有極少數情況（單層
    List[str]，還沒疊過歷史）才需要再包一層。用「第一個元素是不是 list」
    判斷，避免把整個 visit 的代碼清單誤當成單一代碼再包一層、被
    enrich_sample_with_names 的 str() 整包字串化。
    """
    if not val:
        return []
    return val if isinstance(val[0], list) else [val]


def preprocess_samples(dataset):
    """
    重新將 PyHealth 回傳的 record 轉成符合 enrich_sample_with_names
    的格式：所有欄位皆為 List[List[str]]。
    """
    samples = []
    for record in tqdm(dataset, desc="Preprocessing samples"):
        sample = {
            'visit_id':   record['visit_id'],
            'patient_id': record['patient_id'],
            'visit_date':  record.get('visit_date', ''),
            # 用 conditions_all（或原 record['conditions']），只在真的還沒疊過歷史時才包一層
            'conditions': _as_visit_lists(record.get('conditions_all', record['conditions'])),
            # 同 procedures
            'procedures': _as_visit_lists(record.get('procedures_all', record['procedures'])),
            # **重點**：用 drugs_all 而非 record['drugs']
            'drugs':      _as_visit_lists(record.get('drugs_all', record['drugs'])),
            # 若有 label 則取出
            'label':      record.get('label')
        }
        samples.append(sample)
    return samples
