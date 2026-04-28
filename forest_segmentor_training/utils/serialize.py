# from https://github.com/ppwwyyxx/RAM-multiprocess-dataloader/blob/79897b26a2c4185a3ed086f18be5ea300913d5b7/serialize.py
# why? see https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
# https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader
# https://github.com/Lightning-AI/lightning/issues/17257#issuecomment-1493834663

# Copyright (c) Facebook, Inc. and its affiliates.
"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

import pickle
import numpy as np
import torch

class NumpySerializedList:
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)
