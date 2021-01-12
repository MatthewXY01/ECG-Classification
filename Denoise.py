import numpy as np
import pywt

def Denoise(hbs):
    # hbs: (xx, 12)
    # new_hbs = np.empty((hbs.shape[0], hbs.shape[1]))
    for l in range(12):
        noisy = hbs[:, l]
        coeffs = pywt.wavedec(data=noisy, wavelet='db5', level=9)
        cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
        threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1)))) # 将高频信号cD1、cD2置零
        cD1.fill(0)
        cD2.fill(0)
        # 将其他中低频信号按软阈值公式滤波
        for i in range(1, len(coeffs) - 2):
            coeffs[i] = pywt.threshold(coeffs[i], threshold)

        clean = pywt.waverec(coeffs=coeffs, wavelet='db5')
        if clean.size>hbs.shape[0]:
            hbs[:, l] = clean[:hbs.shape[0]]
        else:
            clean.size< hbs.shape[0]
            hbs[:clean.size, l] = clean
