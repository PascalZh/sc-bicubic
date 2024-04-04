# 通过随机计算实现MAC（Multiply-accumulate）操作

import random
import pylfsr
import numpy as np
import tqdm
import os
import matplotlib.pyplot as plt


class LFSR_RNG:

    def __init__(self, inistate='random'):
        self.lfsr = pylfsr.LFSR(fpoly=[23, 18], initstate=inistate)

    def gen_seq(self, seq_len, bit_width=8):
        seq = np.zeros(seq_len, dtype=np.uint32)

        self.lfsr.runKCycle(seq_len * bit_width)

        for i in range(seq_len):
            seq[i] = int(self.lfsr.getSeq()[i * bit_width:i * bit_width +
                                            bit_width],
                         base=2)

        return seq


class CL_RNG:

    def __init__(self, file_name):
        self.file_name = file_name
        self.file_size = os.path.getsize(self.file_name)
        self.f = open(file_name, 'rb')

    def gen_seq(self, seq_len, start=0, bit_width=8):
        assert bit_width % 8 == 0
        seq = np.zeros(seq_len, dtype=np.uint32)

        self.f.seek(start)

        for i in range(seq_len):
            sn = int.from_bytes(self.f.read(bit_width // 8), byteorder='big')
            seq[i] = sn
        return seq

    def __del__(self):
        self.f.close()


def sc_mac4(xs, ws, gen, bit_width=8, seq_len=1024):
    assert len(xs) == len(ws) == 4

    x_seqs = list(
        map(lambda x: gen(round(x * 2**bit_width), seq_len, bit_width), xs))
    w_seqs = list(
        map(lambda w: gen(round(w * 2**bit_width), seq_len, bit_width), ws))

    p_seq = np.zeros(seq_len, dtype=np.uint32)

    r1 = pylfsr.LFSR(fpoly=[23, 18], initstate='random').runKCycle(seq_len)
    r2 = pylfsr.LFSR(fpoly=[23, 18], initstate='random').runKCycle(seq_len)
    r3 = pylfsr.LFSR(fpoly=[23, 18], initstate='random').runKCycle(seq_len)

    m0 = x_seqs[0] & w_seqs[0]
    m1 = x_seqs[1] & w_seqs[1]
    m2 = x_seqs[2] & w_seqs[2]
    m3 = x_seqs[3] & w_seqs[3]

    s0 = np.where(r1, m0, m1)
    s1 = np.where(r2, m2, m3)

    p_seq = np.where(r3, s0, s1)

    return np.sum(p_seq) / seq_len * 4


def sc_mult(A, B, a_seq, b_seq, bit_width=8):
    seq_len = len(a_seq)
    a = a_seq < round(A * 2**bit_width)
    b = b_seq < round(B * 2**bit_width)

    p = np.zeros(seq_len, dtype=np.uint32)

    p = a & b

    return np.sum(p) / seq_len


def sc_add(A, B, a_seq, b_seq, c_seq, bit_width=8):
    seq_len = len(a_seq)

    a = a_seq < round(A * 2**bit_width)
    b = b_seq < round(B * 2**bit_width)
    c = c_seq < round(0.5 * 2**bit_width)

    p = np.where(c, a, b)

    return np.sum(p) / seq_len


def test_mult(N, seq1, seq2, bit_width=8):
    MAE = 0
    MSE = 0
    for _ in range(N):
        A = int(np.random.uniform(0.2, 1) * 2**bit_width) / 2**bit_width
        B = int(np.random.uniform(0.2, 1) * 2**bit_width) / 2**bit_width
        sc_mult_result = sc_mult(A, B, seq1, seq2, bit_width=bit_width)

        mult_result = A * B
        MAE += abs(sc_mult_result - mult_result)
        MSE += (sc_mult_result - mult_result)**2

    MAE /= N
    MSE /= N

    return MAE, MSE


def test_add(N, seq1, seq2, seq3, bit_width=8):
    MAE = 0
    MSE = 0
    for _ in range(N):
        A = int(np.random.uniform(0.2, 1) * 2**bit_width) / 2**bit_width
        B = int(np.random.uniform(0.2, 1) * 2**bit_width) / 2**bit_width
        sc_add_result = sc_add(A, B, seq1, seq2, seq3, bit_width=bit_width)

        add_result = (A + B) / 2

        MAE += abs(sc_add_result - add_result)
        MSE += (sc_add_result - add_result)**2

    MAE /= N
    MSE /= N

    return MAE, MSE


def main():
    N = 10000
    bit_width = 8

    filename = r"D:\0_data\physical_rng\cuibing_processed_CH1tra01.bin"
    # filename = r"D:\0_data\physical_rng\cuibing_raw_tra_01.bin"
    # filename = r"D:\0_data\physical_rng\cuibing_raw_CH1tra01.bin"

    cl_start = np.random.randint(0, 20000)
    cl_start = 10880
    print(f"{cl_start=}")
    cl_seq = CL_RNG(filename).gen_seq(1026,
                                      start=cl_start,
                                      bit_width=bit_width)
    lfsr_seq1 = LFSR_RNG(inistate='random').gen_seq(1024, bit_width=bit_width)
    lfsr_seq2 = LFSR_RNG(inistate='random').gen_seq(1024, bit_width=bit_width)
    lfsr_seq3 = LFSR_RNG(inistate='random').gen_seq(1024, bit_width=bit_width)

    print("testing mult...")
    lfsr_results = []
    chaos_results = []

    for seq_len in [32, 64, 128, 256, 512, 1024]:

        seq1 = lfsr_seq1[:seq_len]
        seq2 = lfsr_seq2[:seq_len]
        lfsr_results.append(test_mult(N, seq1, seq2, bit_width=bit_width))

        seq1 = cl_seq[:seq_len]
        seq2 = cl_seq[1:1 + seq_len]
        chaos_results.append(test_mult(N, seq1, seq2, bit_width=bit_width))

    print(f"lfsr results: {lfsr_results}")
    print(f"chaos results: {chaos_results}")

    plot_results("mult", lfsr_results, chaos_results)

    print("testing add...")
    lfsr_results = []
    chaos_results = []

    # SECTION test add
    for seq_len in [32, 64, 128, 256, 512, 1024]:

        seq1 = lfsr_seq1[:seq_len]
        seq2 = lfsr_seq2[:seq_len]
        seq3 = lfsr_seq3[:seq_len]
        lfsr_results.append(test_add(N, seq1, seq2, seq3, bit_width=bit_width))

        seq1 = cl_seq[:seq_len]
        seq2 = cl_seq[1:1 + seq_len]
        seq3 = cl_seq[2:2 + seq_len]
        chaos_results.append(test_add(N, seq1, seq2, seq3,
                                      bit_width=bit_width))

    print(f"lfsr results: {lfsr_results}")
    print(f"chaos results: {chaos_results}")

    plot_results("add", lfsr_results, chaos_results)

    plt.show()


def plot_results(fig_name, lfsr_results, chaos_results):
    plt.figure(fig_name)
    plt.subplot(121)
    plt.title('MAE')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([32, 64, 128, 256, 512, 1024], [r[0] for r in lfsr_results],
             label='lfsr',
             color='b')
    plt.plot([32, 64, 128, 256, 512, 1024], [r[0] for r in chaos_results],
             label='chaos',
             color='r')
    plt.legend()

    plt.subplot(122)
    plt.title('MSE')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([32, 64, 128, 256, 512, 1024], [r[1] for r in lfsr_results],
             label='lfsr',
             color='b')
    plt.plot([32, 64, 128, 256, 512, 1024], [r[1] for r in chaos_results],
             label='chaos',
             color='r')

    plt.legend()


if __name__ == '__main__':
    main()
