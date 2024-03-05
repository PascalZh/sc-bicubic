# 通过随机计算实现MAC（Multiply-accumulate）操作

import random
import pylfsr
import numpy as np
import tqdm
import os


def gen_rand_seq(x, seq_len, bit_width=8):
    L = pylfsr.LFSR(fpoly=[23, 18], initstate='random')
    L.runKCycle(seq_len * bit_width)
    seq_str = L.getSeq()

    seq = np.zeros(seq_len, dtype=np.uint32)

    for i in range(seq_len):
        sn = int(seq_str[i * bit_width:(i + 1) * bit_width], base=2)
        seq[i] = 1 if sn < x else 0

    return seq


class RandSeqReader:
    start = 0

    def __init__(self, filename):
        self.filename = filename
        self.f = open(filename, 'rb')

    def gen_rand_seq(self, x, seq_len, bit_width=8):
        assert bit_width % 8 == 0
        seq = np.zeros(seq_len, dtype=np.uint32)

        # get file size
        file_size = os.path.getsize(self.filename)
        self.f.seek(self.start)
        self.start = (self.start + 13) % (file_size - seq_len)

        for i in range(seq_len):
            sn = int.from_bytes(self.f.read(bit_width // 8), byteorder='big')
            seq[i] = 1 if sn < x else 0
        return seq

    def __del__(self):
        self.f.close()


def sc_mac(xs, ws, gen_rand_seq):
    seq_len = 1024
    bit_width = 10
    x_seqs = list(
        map(lambda x: gen_rand_seq(int(x * 2**bit_width), seq_len, bit_width),
            xs))
    w_seqs = list(
        map(lambda w: gen_rand_seq(int(w * 2**bit_width), seq_len, bit_width),
            ws))

    p_seq = np.zeros(seq_len, dtype=np.uint32)

    r1 = pylfsr.LFSR(fpoly=[23, 18], initstate='random').runKCycle(seq_len)
    r2 = pylfsr.LFSR(fpoly=[23, 18], initstate='random').runKCycle(seq_len)
    r3 = pylfsr.LFSR(fpoly=[23, 18], initstate='random').runKCycle(seq_len)

    for i in range(seq_len):
        m0 = x_seqs[0][i] & w_seqs[0][i]
        m1 = x_seqs[1][i] & w_seqs[1][i]
        m2 = x_seqs[2][i] & w_seqs[2][i]
        m3 = x_seqs[3][i] & w_seqs[3][i]

        s0 = m0 if r1[i] else m1
        s1 = m2 if r2[i] else m3

        p_seq[i] = s0 if r3[i] else s1

    return np.sum(p_seq) / seq_len * 4


def sc_mult(a, b, gen, bit_width=8, seq_len=1024):
    a_seq = gen(int(a * 2**bit_width), seq_len, bit_width)
    b_seq = gen(int(b * 2**bit_width), seq_len, bit_width)

    p_seq = np.zeros(seq_len, dtype=np.uint32)

    p_seq = a_seq & b_seq

    return np.sum(p_seq) / seq_len


def test_mac():
    MAE = 0
    N = 10
    for _ in tqdm.tqdm(range(N)):
        xs = np.random.uniform(0, 1, 4)
        ws = np.random.uniform(0, 1, 4)
        sc_mac_result = sc_mac(xs, ws, gen_rand_seq)
        mac_result = np.dot(xs, ws)
        MAE += abs(sc_mac_result - mac_result)
    print(f"MAE={MAE / N:.3f}")


def test_mult(N, gen):
    MAE = 0
    bit_width = 8
    for _ in tqdm.tqdm(range(N)):
        a = np.random.uniform(0, 1)
        b = np.random.uniform(0, 1)
        sc_mult_result = sc_mult(a, b, gen, bit_width=bit_width, seq_len=1024)

        mult_result = int(a * 2**bit_width) * int(
            b * 2**bit_width) / 2**(bit_width * 2)
        MAE += abs(sc_mult_result - mult_result)

    print(f"MAE={MAE / N:.3f}")


def main():
    N = 1000

    filename = r"D:\0_data\physical_rng\cuibing_processed_CH1tra01.bin"
    # filename = r"D:\0_data\physical_rng\cuibing_raw_tra_01.bin"
    # filename = r"D:\0_data\physical_rng\cuibing_raw_CH1tra01.bin"

    test_mult(N, gen_rand_seq)
    test_mult(N, RandSeqReader(filename).gen_rand_seq)


if __name__ == '__main__':
    main()
