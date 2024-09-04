import numpy as np
from typing import List

class Bucket:
    # 这里的max_seqlen表示的是显存支持的最大值
    def __init__(self, pad_token: int, max_seqlen: int):
        self._pad_token = pad_token
        self._max_seqlen = max_seqlen
        self._batch = []
        self._cu_seqlens_list = []
        self._packed_batch = None
        self._packed_cu_seqlens_list = None

    def add_data(self, padded_sequence, valid_tokens):
        self._batch.append(padded_sequence[:valid_tokens])
        self._cu_seqlens_list.append(np.array([0, valid_tokens], dtype=np.int32))

    # 简单的贪心packing策略
    # 已经默认batch中的数据按照从短到长排序
    def pack_data(self):
        is_visited = set()
        packed_batch = []
        packed_cu_seqlens_list = []
        for i in range(len(self._batch)):
            if i in is_visited:
                continue
            packed_seqs = [self._batch[i]]
            cur_cu_seqlen = len(self._batch[i])
            cu_seqlens = [0, cur_cu_seqlen]
            is_visited.add(i)
            for j in reversed(range(i + 1, len(self._batch))):
                if j not in is_visited and cur_cu_seqlen + len(self._batch[j]) <= self._max_seqlen:
                    packed_seqs.append(self._batch[j])
                    cur_cu_seqlen += len(self._batch[j])
                    cu_seqlens.append(cur_cu_seqlen)
                    is_visited.add(j)
            # 目前先统一pad到max_seqlen
            if cur_cu_seqlen < self._max_seqlen:
                packed_seqs.append(np.array([self._pad_token] * (self._max_seqlen - cu_seqlen)))
            packed_batch.append(np.concatenate(packed_seqs))
            packed_cu_seqlens_list.append(np.array(cu_seqlens, dtype=np.int32))
        assert len(packed_batch) > 0, "currently not support no data after packing"
        self._packed_batch = packed_batch
        self._packed_cu_seqlens_list = packed_cu_seqlens_list

    def packed_batch_size(self):
        assert self._packed_batch != None, "please ensure you have packed the bucket"
        return len(self._packed_batch)

    def original_batch_size(self):
        return len(self._batch)

    def packed_batch(self):
        assert self._packed_batch != None, "please ensure you have packed the bucket"
        return self._packed_batch
    
    def packed_cu_seqlens_list(self):
        assert self._packed_cu_seqlens_list != None, "please ensure you have packed the bucket"
        return self._packed_cu_seqlens_list 
 
# 对global batch中的seq按从小到大的顺序进行排序   
def get_sorted_batch_and_len(global_batch: np.ndarray, pad_token: int):
    non_pad_counts = np.sum(global_batch != pad_token, axis=1)
    sorted_indices = np.argsort(-non_pad_counts)
    sorted_global_batch = global_batch[sorted_indices]
    sorted_valid_tokens = non_pad_counts[sorted_indices]
    return sorted_global_batch, sorted_valid_tokens

# 从global batch中取出batch_indices的seq构成一个bucket
def get_bucket(global_batch: np.ndarray, pad_token: int, batch_indices: List[int], max_seqlen: int):
    bucket_batch = global_batch[bucket_indices]
    bucket_valid_tokens = np.sum(bucket_batch != pad_token, axis=1)
    bucket = Bucket(pad_token, max_seqlen)
    for seq, vailid_tokens in zip(bucket_batch, bucket_valid_tokens):
        bucket.add_data(seq, vailid_tokens)
    return bucket