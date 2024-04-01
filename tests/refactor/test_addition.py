import hetu
import hetu.nn as nn
import torch.optim as optim
import numpy as np
import torch
import unittest
from test_utils import allclose
import os
import sys
import random

NUM_ITERS = 20

# Warning: Remember to set rtol = 1e-05, atol = 3e-05 in `test_utils.py`

class TestOtherOps(unittest.TestCase):
    
    _attn_shapes = [
        ((2, 128, 8, 16),),
    ]

    # def test_attnop(self):
    #     for iter in range(NUM_ITERS):
    #         batch_size = 4
    #         num_heads = 8
    #         head_size = 16
    #         seq_len = [(1 << a) for a in range(7, 14)]
    #         # print(seq_len)
    #         weights = [1.0] * 7
    #         weights[0] = 0.206492
    #         weights[1] = 0.228799
    #         weights[2] = 0.233064
    #         weights[3] = 0.197762
    #         weights[4] = 0.090700
    #         weights[5] = 0.029225
    #         for i in range(6):
    #             weights[6] -= weights[i]
    #         # print(weights)
    #         batch_seq_len = random.choices(seq_len, weights=weights, k=batch_size) 
    #         # print(batch_seq_len)
    #         max_len = max(batch_seq_len)
    #         sum_len = sum(batch_seq_len)
    #         padding_shape = [batch_size, max_len, num_heads, head_size]
    #         varlen_shape = [sum_len, num_heads, head_size]
    #         q1_np = np.random.randn(*padding_shape).astype(np.float32)
    #         k1_np = np.random.randn(*padding_shape).astype(np.float32)
    #         v1_np = np.random.randn(*padding_shape).astype(np.float32)
    #         q2_np = np.random.randn(*varlen_shape).astype(np.float32)
    #         k2_np = np.random.randn(*varlen_shape).astype(np.float32)
    #         v2_np = np.random.randn(*varlen_shape).astype(np.float32)
    #         batch_seq_len.append(batch_size)
    #         len_q_np = np.array(batch_seq_len, dtype=np.int32)
    #         len_k_np = np.array(batch_seq_len, dtype=np.int32)
    #         # print(q1_np.shape, ",", q2_np.shape, ",", len_q_np)

    #         q1 = hetu.from_numpy(q1_np).to(dtype=hetu.bfloat16)
    #         k1 = hetu.from_numpy(k1_np).to(dtype=hetu.bfloat16)
    #         v1 = hetu.from_numpy(v1_np).to(dtype=hetu.bfloat16)

    #         q2 = hetu.from_numpy(q2_np).to(dtype=hetu.bfloat16)
    #         k2 = hetu.from_numpy(k2_np).to(dtype=hetu.bfloat16)
    #         v2 = hetu.from_numpy(v2_np).to(dtype=hetu.bfloat16)

    #         len_q = hetu.from_numpy(len_q_np)
    #         len_k = hetu.from_numpy(len_k_np)
            
    #         # attn1 = hetu.attn(q1, k1, v1)[0]
    #         attn2 = hetu.attn_varlen(q2, k2, v2, len_q, len_k, 8192, 8192)[0]
    #     # for iter in range(NUM_ITERS):
    #     #     batch_size = 256
    #     #     num_heads = 8
    #     #     head_size = 16
    #     #     seq_len = [(1 << a) for a in range(7, 14)]
    #     #     # print(seq_len)
    #     #     weights = [1.0] * 7
    #     #     weights[0] = 0.206492
    #     #     weights[1] = 0.228799
    #     #     weights[2] = 0.233064
    #     #     weights[3] = 0.197762
    #     #     weights[4] = 0.090700
    #     #     weights[5] = 0.029225
    #     #     for i in range(6):
    #     #         weights[6] -= weights[i]
    #     #     # print(weights)
    #     #     batch_seq_len = random.choices(seq_len, weights=weights, k=batch_size) 
    #     #     # print(batch_seq_len)
    #     #     max_len = max(batch_seq_len)
    #     #     sum_len = sum(batch_seq_len)
    #     #     v1 = 0
    #     #     v2 = 0
    #     #     v3 = 0
    #     #     for i in range(int(batch_size) // 2):
    #     #         v1 += 2 * max(batch_seq_len[2*i], batch_seq_len[2*i+1])
    #     #     sorted_list = sorted(batch_seq_len)
    #     #     for i in range(int(batch_size) // 2):
    #     #         v2 += 2 * max(sorted_list[2*i], sorted_list[2*i+1])
    #     #     for i in range(int(batch_size) // 2):
    #     #         v3 += batch_seq_len[2*i] + batch_seq_len[2*i+1]
    #     #     print("v1-v2-v3:", v1,"-", v2, "-", v3)


    matmul_shapes = [
        ((64, 256), (256, 128)),
        ((128, 256), (256, 256)),
        ((256, 512), (512, 256)),
        ((512, 1024), (1024, 512)),
        ((1024, 1024), (1024, 1024)),
        ((2048, 1024), (1024, 2048)),
        ((4096, 1024), (1024, 4096)),
        ((8192, 1024), (1024, 8192)),
        ((1, 128), (128, 512)),
        ((1, 256), (256, 1024)),
        ((1, 512), (512, 2048)),
        ((1, 1024), (1024, 4096)),
        ((1, 2048), (2048, 8192)),
        ((1, 4096), (4096, 16384)),
        ((4, 1024), (1024, 4096)),
        ((8, 1024), (1024, 4096)),
        ((16, 1024), (1024, 4096)),
        # ((4, 128), (128, 32)),
    ]

    def test_matmul4bit(self):
        # for i in range(10):
        #     x_np = np.random.randn(32, 32, 32).astype(np.float32)
        #     x = hetu.from_numpy(x_np).to(hetu.bfloat16)
        #     x = hetu.add(x, 33)

        for shape_x, shape_y in TestOtherOps.matmul_shapes:
            import math
            x_np = np.random.randn(*shape_x).astype(np.float32)
            y_np = np.random.randn(*shape_y).astype(np.float32)/math.sqrt(shape_x[1])
            nf4_data = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                        -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                        0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                        0.7229568362236023, 1.0]
            code_np = np.array(nf4_data).astype(np.float32)
            absmax_np = np.random.randn(int(shape_x[0] * shape_y[1] / 64)).astype(np.float32)
            x = hetu.from_numpy(x_np).to(dtype=hetu.float32)
            y = hetu.from_numpy(y_np).to(dtype=hetu.float32)
            x_t = torch.from_numpy(x_np).to("cuda:0")
            y_t = torch.from_numpy(y_np).to("cuda:0")
            y_T = y_t.t().contiguous()
            code = hetu.from_numpy(code_np)
            out = hetu.quantization(y, hetu.nfloat4, 64)
            y1 = out[0]
            absmax = out[1]

            # for i in range(10):
            #     z1 = hetu.matmul(x, y)
            #     # z2 = hetu.matmul4bit(x, y1, absmax, code, False, False, 64)
            
            # times = 0
            # for i in range(20):
            #     z1 = hetu.matmul(x, y)
            #     times += z1.timecost() / 1000.0
            # print("Matmul with shape:", shape_x, shape_y, ",cost:",  (times / 20),"ms")
            # times = 0
            # for i in range(20):
            #     z2 = hetu.matmul4bit(x, y1, absmax, code, False, False, 64)
            #     times += z2.timecost() / 1000.0
            # print("Matmul4bit with shape:", shape_x, shape_y, ",cost:",  (times / 20),"ms")
            z1 = hetu.matmul(x, y)
            z2 = hetu.matmul4bit(x, y1, absmax, code, False, False, 64)
            y_ = hetu.dequantization(y1, absmax, hetu.float32, 64)
            z3 = hetu.matmul(x, y_)
            from bitsandbytes import functional as F
            if shape_x[0] == 1:
                qB, state = F.quantize_4bit(y_t, quant_type='nf4', compress_statistics=False)
                z4 = F.gemv_4bit(x_t, qB, state=state)
                y11 = F.dequantize_4bit(qB, quant_state = state, absmax = state.absmax, blocksize = 64, quant_type='nf4')
                z5 = torch.matmul(x_t, y_t)
                qB, state = F.quantize_4bit(y_T, quant_type='nf4', compress_statistics=False)
                print("absmax_t:", state.absmax.cpu().detach().numpy().flatten()[:64])
                z6 = F.gemv_4bit(x_t, qB.t(), state=state)
                y12 = F.dequantize_4bit(qB, quant_state = state, absmax = state.absmax, blocksize = 64, quant_type='nf4').t().contiguous()
            y_T = y.transpose([1, 0]).contiguous()
            out = hetu.quantization(y_T, hetu.nfloat4, 64)
            y1 = out[0]
            absmax = out[1]
            y_2 = hetu.dequantization(y1, absmax, hetu.float32, 64).transpose([1,0]).contiguous()
            print("absmax_h:", absmax.numpy(force=True).flatten()[:64])
            z7 = hetu.matmul4bit(x, y1, absmax, code, False, True, 64)
            if shape_x[0] == 1:
                z8 = torch.matmul(x_t, y11)
            print("Y:",y.numpy(force=True).flatten()[:64])
            # print("Y1:",y1.numpy(force=True).flatten()[:10])
            print("Y_:",y_.numpy(force=True).flatten()[:64])
            print("Y_2:",y_2.numpy(force=True).flatten()[:64])
            if shape_x[0] == 1:
                print("Y_3:",y11.cpu().detach().numpy().flatten()[:64])
                print("Y_4:",y12.cpu().detach().numpy().flatten()[:64])
            print("Z1:",z1.numpy(force=True).flatten()[:64])
            print("Z2:",z2.numpy(force=True).flatten()[:10])
            print("Z3:",z3.numpy(force=True).flatten()[:10])
            if shape_x[0] == 1:
                print("Z4:",z4.cpu().detach().numpy().flatten()[:10])
                print("Z5:",z5.cpu().detach().numpy().flatten()[:10])
                print("Z6:",z6.cpu().detach().numpy().flatten()[:10])
            print("Z7:",z7.numpy(force=True).flatten()[:64])
            if shape_x[0] == 1:
                print("Z8:",z8.cpu().detach().numpy().flatten()[:10])
        
            
                

if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    with hetu.graph("eager"):
        with hetu.context(eager_device="cuda:0"):
            unittest.main()
