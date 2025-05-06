import os
import random
import time

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from IPython.display import Image, display

from raycast import get_raycast


def imshow(img, ax=None):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.ndim == 3:
            img = np.transpose(img, (1, 2, 0))  # CxHxW → HxWxC
    if ax is None:
        ret, encoded = cv2.imencode(".jpg", (img * 255).astype(np.uint8))
        display(Image(encoded))
    else:
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.axis('off')


def to_tensor_cuda(image):
    tensor = torch.from_numpy(image).float() / 255.0
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)  # Add channel dim for grayscale
    else:
        tensor = tensor.permute(2, 0, 1)  # HWC to CHW
    return tensor.cuda()


def to_numpy_img(tensor):
    return (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def main(image_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start = time.time()
    img = cv2.imread(image_path)
    img_t = to_tensor_cuda(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_t = to_tensor_cuda(gray)

    # Thresholding (Otsu on CPU)
    _, bin_img_np = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bin_img = to_tensor_cuda(bin_img_np)

    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
    bin_img_4d = bin_img.unsqueeze(0)

    # Morphology Open
    bin_img_open = torch.nn.functional.conv2d(bin_img_4d, kernel, padding=1)
    bin_img_open = (bin_img_open == kernel.sum()).float()
    bin_img = bin_img_open.squeeze(0)

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 8))

    # Dilation (Sure background)
    dilated = torch.nn.functional.max_pool2d(bin_img.unsqueeze(0), 3, stride=1, padding=1)
    sure_bg = dilated.squeeze(0)
    imshow(sure_bg, axes[0, 0])
    axes[0, 0].set_title('Sure Background')

    # Distance transform (OpenCV fallback)
    bin_img_np = (bin_img.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    dist = cv2.distanceTransform(bin_img_np, cv2.DIST_L2, 5)
    dist_t = to_tensor_cuda(dist)
    imshow(dist_t, axes[0, 1])
    axes[0, 1].set_title('Distance Transform')

    # Threshold sure foreground
    sure_fg = (dist_t > 0.5 * dist_t.max()).float()
    imshow(sure_fg, axes[1, 0])
    axes[1, 0].set_title('Sure Foreground')

    # Unknown region
    unknown = sure_bg - sure_fg
    imshow(unknown, axes[1, 1])
    axes[1, 1].set_title('Unknown')

    # Watershed – fallback to CPU
    markers_np = np.uint8(sure_fg.cpu().numpy() * 255)
    # Make sure markers_np is single-channel uint8
    markers_np = (sure_fg.squeeze().detach().cpu().numpy() * 255).astype(np.uint8)
    num_labels, markers = cv2.connectedComponents(markers_np)
    markers += 1
    unknown_np = unknown.squeeze().detach().cpu().numpy()
    markers[unknown_np > 0.5] = 0
    markers = cv2.watershed(img, markers)

    axes[0, 2].set_title('Zone')
    imshow(markers, axes[0, 2])

    labels = np.unique(markers)
    coins = []
    for label in labels[2:]:
        target = np.where(markers == label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            coins.append(contours[0])

    img_contours = img.copy()
    cv2.drawContours(img_contours, coins, -1, color=(0, 23, 223), thickness=2)
    axes[1, 2].set_title('With contours')
    imshow(img_contours, axes[1, 2])

    # Create mask
    black_img = np.zeros_like(img_contours, dtype="uint8")
    cv2.drawContours(black_img, coins, -1, color=(255, 255, 255), thickness=2)
    black_img[-1, :] = 0
    black_img[-2, :] = 0
    black_img[-3, :] = 0
    axes[1, 3].set_title('Mask')
    imshow(black_img, axes[1, 3])

    # Raycast
    get_raycast(black_img, 10, 180)
    axes[0, 3].set_title('With Raycast')
    imshow(black_img, axes[0, 3])

    print(f'did opencv + torch in {time.time() - start:.2f}s')

    plt.tight_layout()
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    NB_IMAGES_PER_DIR = 10
    DIRS_PATH = [
        '../car_pictures/256_128',
        '../car_pictures/320_180'
    ]
    for DIR in DIRS_PATH:
        images = os.listdir(DIR)
        for image in random.choices(images, k=NB_IMAGES_PER_DIR):
            main(os.path.join(DIR, image))
