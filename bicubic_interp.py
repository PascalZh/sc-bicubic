import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import trange
from functools import lru_cache
from sc_mac import CL_RNG, LFSR_RNG

filename = r"D:\0_data\physical_rng\cuibing_processed_CH1tra01.bin"

bit_width = 8
bsl = 64

# cl_start = np.random.randint(0, 20000)
cl_start = 0
print(f"{cl_start=}")
rand_seq = CL_RNG(filename).gen_seq(bsl * 8,
                                    start=cl_start,
                                    bit_width=bit_width)
# rand_seq = LFSR_RNG().gen_seq(B * 8, bit_width=bit_width)

errors = []


@lru_cache(maxsize=4)
def weight(x):
    t = (2 * x + 1) / 8
    return 0.5 * np.array([1, t, t**2, t**3]) @ np.array(
        [[0, 2, 0, 0], [-1, 0, 1, 0], [2, -5, 4, -1], [-1, 3, -3, 1]])


# weights = np.array([weight(x) for x in range(4)])
# plt.hist(weights.flatten(), bins=100)
# plt.show()


def cubic_interpolate(x, f0, f1, f2, f3):
    # return np.dot(weight((x - 2) % 4), np.array([f0, f1, f2, f3]))

    w0, w1, w2, w3 = weight((x - 2) % 4)
    # print(f"{w0=}, {w1=}, {w2=}, {w3=}")

    n = 4

    # take high n bits of f
    fh0 = f0 >> (bit_width - n)
    fh1 = f1 >> (bit_width - n)
    fh2 = f2 >> (bit_width - n)
    fh3 = f3 >> (bit_width - n)

    fl0 = f0 & ((1 << (bit_width - n)) - 1)
    fl1 = f1 & ((1 << (bit_width - n)) - 1)
    fl2 = f2 & ((1 << (bit_width - n)) - 1)
    fl3 = f3 & ((1 << (bit_width - n)) - 1)

    h = 2**(8 - n)
    sc_l = sc_mac(fl0 / h, fl1 / h, fl2 / h, fl3 / h, w0, w1, w2, w3) * h
    sc_p = 2**(8 - n) * (fh0 * w0 + fh1 * w1 + fh2 * w2 + fh3 * w3) + sc_l

    # sc_p = sc_mac(f0 / 255, f1 / 255, f2 / 255, f3 / 255, w0, w1, w2, w3) * 255

    # p = np.dot(weight((x - 2) % 4), np.array([f0, f1, f2, f3]))
    # true_l = fl0 * w0 + fl1 * w1 + fl2 * w2 + fl3 * w3
    # errors.append(true_l - sc_l)
    # if len(errors) % 1000 == 0:
    #     MSE = np.mean(np.array(errors)**2)
    #     PSNR = 10 * np.log10(255**2 / MSE)
    #     plt.hist(errors, bins=100)
    #     plt.show()
    return sc_p


def sc_mac(f0, f1, f2, f3, w0, w1, w2, w3):
    # assert 0 <= f0 <= 1 and 0 <= f1 <= 1 and 0 <= f2 <= 1 and 0 <= f3 <= 1
    # assert -1 <= w0 <= 1 and -1 <= w1 <= 1 and -1 <= w2 <= 1 and -1 <= w3 <= 1
    W0 = rand_seq[:bsl] < int((w0 + 1) / 2 * 2**bit_width)
    W1 = rand_seq[bsl:2 * bsl] < int((w1 + 1) / 2 * 2**bit_width)
    W2 = rand_seq[2 * bsl:3 * bsl] < int((w2 + 1) / 2 * 2**bit_width)
    W3 = rand_seq[3 * bsl:4 * bsl] < int((w3 + 1) / 2 * 2**bit_width)

    F0 = rand_seq[4 * bsl:5 * bsl] < int((f0 + 1) / 2 * 2**bit_width)
    F1 = rand_seq[5 * bsl:6 * bsl] < int((f1 + 1) / 2 * 2**bit_width)
    F2 = rand_seq[6 * bsl:7 * bsl] < int((f2 + 1) / 2 * 2**bit_width)
    F3 = rand_seq[7 * bsl:8 * bsl] < int((f3 + 1) / 2 * 2**bit_width)

    # C0 = cl_seq[8000:8000 + seq_len] < int(0.5 * 255)
    # C1 = cl_seq[9000:9000 + seq_len] < int(0.5 * 255)
    # C2 = cl_seq[10000:10000 + seq_len] < int(0.5 * 255)

    M0 = ~(W0 ^ F0)
    M1 = ~(W1 ^ F1)
    M2 = ~(W2 ^ F2)
    M3 = ~(W3 ^ F3)

    # S0 = np.where(C0, M0, M1)
    # S1 = np.where(C1, M2, M3)
    # S2 = np.where(C2, S0, S1)

    def tff_adder(A, B):
        T = A ^ B
        Q = np.insert(T[:-1], 0, 0)
        return np.where(T, Q, B)

    S0 = tff_adder(M0, M1)
    S1 = tff_adder(M2, M3)
    S2 = tff_adder(S0, S1)

    n1 = np.sum(S2)
    n0 = bsl - n1

    sc_p = (n1 - n0) / bsl * 4
    return sc_p


def upscale_bicubic(data_in, data_out):

    dx = dy = ch = 0  # dx, dy表示目标图像的坐标，ch表示彩色通道
    sx = sy = 0  # sx, sy表示原图像的坐标
    sxk = syk = 0
    # sxk, syk分别用于遍历[sx-1, sx, sx+1, sx+2]和[sy-1, sy, sy+1, sy+2]
    b = np.zeros((4, 3), dtype=np.uint32)  # 用于暂存x方向插值的结果，所以有四个，每个有3个通道

    height_in, width_in, _ = data_in.shape
    height_out, width_out, _ = data_out.shape

    for dy in trange(height_out):
        for dx in range(width_out):
            sx = (dx - 2) // 4
            sy = (dy - 2) // 4
            if dx < 2:
                sx = -1
            if dy < 2:
                sy = -1

            # 对原图像坐标范围为[sx-1, sx, sx+1, sx+2]x[sy-1, sy, sy+1, sy+2]的4x4小区域进行插值
            for i in range(4):
                fs = np.zeros((4, 3), dtype=np.uint32)
                for j in range(4):
                    sxk = sx + j - 1
                    syk = sy + i - 1
                    # 这一步操作相当于padding，使用边界值填充
                    if sxk < 0:
                        sxk = 0
                    if sxk >= width_in:
                        sxk = width_in - 1
                    if syk < 0:
                        syk = 0
                    if syk >= height_in:
                        syk = height_in - 1
                    for ch in range(3):
                        fs[j, ch] = data_in[syk, sxk, ch]
                for ch in range(3):
                    b[i, ch] = np.clip(
                        cubic_interpolate(dx, fs[0, ch], fs[1, ch], fs[2, ch],
                                          fs[3, ch]), 0, 255)

            for ch in range(3):
                p = cubic_interpolate(dy, b[0, ch], b[1, ch], b[2, ch], b[3,
                                                                          ch])
                data_out[dy, dx, ch] = np.clip(p, 0, 255)


def main():
    img = cv2.imread('Lena.bmp')
    data_in = cv2.resize(img, (128, 128))
    data_out = np.zeros((512, 512, 3), dtype=np.uint8)

    opencv_sr = cv2.resize(data_in, (512, 512), interpolation=cv2.INTER_CUBIC)

    upscale_bicubic(data_in, data_out)

    cv2.imshow("HR", img)
    cv2.imshow("LR", data_in)
    cv2.imshow("sc bicubic interpolation", data_out)
    cv2.imwrite("sc_bicubic_sr.bmp", data_out)
    cv2.imshow("opencv bicubic interpolation", opencv_sr)
    cv2.waitKey(0)


def analyze():
    img = cv2.imread('Lena.bmp')
    lr = cv2.resize(img, (128, 128))

    sc_bicubic_sr = cv2.imread("sc_bicubic_sr.bmp")
    opencv_bicubic_sr = cv2.resize(lr, (512, 512),
                                   interpolation=cv2.INTER_CUBIC)

    err_img = np.abs(sc_bicubic_sr - img)
    MSE = np.mean(err_img**2)
    PSNR = 10 * np.log10(255**2 / MSE)
    print(f"sc bicubic PSNR: {PSNR} dB")

    MSE = np.mean((opencv_bicubic_sr - img)**2)
    PSNR = 10 * np.log10(255**2 / MSE)
    print(f"opencv bicubic PSNR: {PSNR} dB")

    cv2.imshow("sc bicubic interpolation", sc_bicubic_sr)
    cv2.imshow("opencv bicubic interpolation", opencv_bicubic_sr)
    cv2.imshow("error", err_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    # main()
    # analyze()


    # test sc_mac
    # for _ in range(10000):
    #     As = np.random.uniform(0.2, 1, 4)
    #     Bs = np.random.uniform(0.2, 1, 4)
    #     sc_result = sc_mac(As[0], As[1], As[2], As[3], Bs[0], Bs[1], Bs[2],
    #                        Bs[3])
    #     true_result = np.sum(As * Bs)

    #     print(f"{sc_result=}, {true_result=}, {sc_result - true_result}")
