import os
import random
import time

import cv2
import numpy as np
from IPython.display import Image, display
from matplotlib import pyplot as plt

from raycast import get_raycast


def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv2.imencode(".jpg", img)
        display(Image(encoded))
    else:
        ax.imshow(img)
        ax.axis('off')


def main(image_path):
    start = time.time()
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 8))
    sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
    imshow(sure_bg, axes[0, 0])
    axes[0, 0].set_title('Sure Background')
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    imshow(dist, axes[0, 1])
    axes[0, 1].set_title('Distance Transform')
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    imshow(sure_fg, axes[1, 0])
    axes[1, 0].set_title('Sure Foreground')
    unknown = cv2.subtract(sure_bg, sure_fg)
    imshow(unknown, axes[1, 1])
    axes[1, 1].set_title('Unknown')
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    axes[0, 2].set_title('Zone')
    imshow(markers, axes[0, 2])
    labels = np.unique(markers)
    coins = []
    for label in labels[2:]:
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        coins.append(contours[0])
    img = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)
    axes[1, 2].set_title('With contours')
    imshow(img, axes[1, 2])

    black_img = np.zeros(img.shape, dtype="uint8")
    black_img = cv2.drawContours(black_img, coins, -1, color=(255, 255, 255), thickness=2)
    axes[1, 3].set_title('Mask')
    black_img[:][-1] = [0, 0, 0]
    black_img[:][-2] = [0, 0, 0]
    black_img[:][-3] = [0, 0, 0]
    imshow(black_img, axes[1, 3])

    get_raycast(black_img, 10, 180)
    axes[0, 3].set_title('With Raycast')
    imshow(black_img, axes[0, 3])
    print(f'did opencv in {time.time() - start}')

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    NB_IMAGES_PER_DIR = 1
    DIRS_PATH = [
        '../car_pictures/256_128',
        '../car_pictures/320_180'
    ]
    for DIR in DIRS_PATH:
        images = os.listdir(DIR)
        for image in random.choices(images, k=NB_IMAGES_PER_DIR):
            main(os.path.join(DIR, image))
