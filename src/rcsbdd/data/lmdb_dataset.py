import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset


class CrossDockedLMDB(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.env = lmdb.open(str(self.path), readonly=True, lock=False, readahead=False, max_readers=32)
        with self.env.begin() as txn:
            self.keys = [k for k, _ in txn.cursor() if k != b"__meta__"]
            meta = txn.get(b"__meta__")
            self.meta = _load_meta(meta) if meta is not None else {}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with self.env.begin() as txn:
            item = pickle.loads(txn.get(key))
        return item


def collate_basic(items):
    batch = {
        "lig_n": torch.tensor([x["lig_pos"].shape[0] for x in items], dtype=torch.float32),
        "pock_n": torch.tensor([x["pock_pos"].shape[0] for x in items], dtype=torch.float32),
        "lig_pos_mean": torch.tensor(np.stack([x["lig_pos"].mean(axis=0) for x in items]), dtype=torch.float32),
        "pock_pos_mean": torch.tensor(np.stack([x["pock_pos"].mean(axis=0) for x in items]), dtype=torch.float32),
        "pock_feat_mean": torch.tensor(np.stack([x["pock_feat"].mean(axis=0) for x in items]), dtype=torch.float32),
        "names": [x.get("name", "") for x in items],
    }
    return batch


def _load_meta(blob):
    try:
        return pickle.loads(blob)
    except Exception:
        return json.loads(blob.decode("utf-8"))
