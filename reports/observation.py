from pathlib import Path
import os
import cv2
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

src = Path().absolute().parent.joinpath('data/raw/train/')

imgs = src.joinpath('images').iterdir()
masks = src.joinpath('masks').iterdir()

for img, mask in zip(imgs, masks):

    img = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((5, 5), np.float32) / 25
    dst = cv2.filter2D(img, -1, kernel)

    # norm_img = stats.zscore(img, axis=None)
    # plt.subplot(121).imshow(img)
    plt.subplot(121).imshow(dst, cmap='seismic')
    plt.subplot(122).imshow(mask, cmap='seismic')
    plt.show()
    plt.close()

