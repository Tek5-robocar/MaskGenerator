import os
import random
import time
import math

import cv2
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import Pad
import numpy as np
from scipy.spatial import distance
from Dataset import TestDataset

from UNet_3Plus import UNet_3Plus
from models.fast_scnn import FastSCNN
from Unet import UNet
from EfficientLiteSeg import EfficientLiteSeg


def get_white_contours(bw_image):
    if len(bw_image.shape) > 2:
        raise ValueError("Image should be single-channel black and white")
    contours, _ = cv2.findContours(bw_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_closest_points(contour1, contour2):
    contour1 = contour1.reshape(-1, 2)
    contour2 = contour2.reshape(-1, 2)

    min_dist = float('inf')
    closest_pt1 = None
    closest_pt2 = None

    for pt1 in contour1:
        for pt2 in contour2:
            dist = distance.euclidean(pt1, pt2)
            if dist < min_dist:
                min_dist = dist
                closest_pt1 = pt1
                closest_pt2 = pt2

    return closest_pt1, closest_pt2, min_dist


def get_two_closest_contours(target_contour, all_contours):
    if len(all_contours) < 3:
        raise ValueError("Need at least 3 contours to find two closest (excluding target)")

    first_dist = second_dist = float('inf')
    first_closest = second_closest = None
    first_points = second_points = (None, None)

    target_flat = target_contour.reshape(-1, 2)

    for contour in all_contours:
        if np.array_equal(contour, target_contour):
            continue

        contour_flat = contour.reshape(-1, 2)
        min_dist = float('inf')
        current_points = (None, None)

        for pt1 in target_flat:
            for pt2 in contour_flat:
                dist = distance.euclidean(pt1, pt2)
                if dist < min_dist:
                    min_dist = dist
                    current_points = (pt1, pt2)

        if min_dist < first_dist:
            second_dist = first_dist
            second_closest = first_closest
            second_points = first_points

            first_dist = min_dist
            first_closest = contour
            first_points = current_points
        elif min_dist < second_dist:
            second_dist = min_dist
            second_closest = contour
            second_points = current_points

    if first_closest is None or second_closest is None:
        raise ValueError("Couldn't find two distinct contours (duplicates?)")

    return (first_closest, first_dist, first_points,
            second_closest, second_dist, second_points)


def clean_mask(mask):
    contours = get_white_contours(mask.astype(np.uint8))
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i in range(len(contours) - 1):
        pt1 = tuple(contours[i][0, 0])
        try:
            first_closest, first_dist, first_points, second_closest, second_dist, second_points = get_two_closest_contours(
                contours[i], contours)
        except ValueError:
            return mask, mask
        cv2.line(output, pt1, first_points[1], (255, 255, 255), 2)
        cv2.line(output, pt1, second_points[1], (255, 255, 255), 2)

    output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    pred = (mask * 255).astype(np.uint8)
    merged = cv2.bitwise_or(pred, output)
    return output, merged


def overlay_mask_on_image(image, mask):
    image_rgb = image.copy()
    if len(image_rgb.shape) == 2 or image_rgb.shape[2] == 1:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

    mask = (mask * 255).astype(np.uint8)
    if len(mask.shape) > 2:
        mask = mask.squeeze()

    # Directly replace masked pixels with pure red
    overlaid_image = image_rgb.copy()
    overlaid_image[mask == 255] = [255, 0, 0]  # Fully opaque red

    return overlaid_image


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    output_dir = 'benchmark_results'
    os.makedirs(output_dir, exist_ok=True)

    models = [
        ('Unet', UNet(in_channels=3, out_channels=1).to(device)),
        ('EfficientLiteSeg', EfficientLiteSeg(in_channels=3, out_channels=1).to(device)),
    ]

    transform_img = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    for sub_dir in SUB_DIRS:
        dataset_test = TestDataset(os.path.join(DIR, sub_dir), transform_img=transform_img)
        indices = random.sample(range(len(dataset_test)), NB_IMAGES_PER_DIRS)

        for indice in indices:
            image = dataset_test[indice]
            img_input = image.unsqueeze(0).to(device)

            img_np = image.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            total_preds = sum(len(os.listdir(f'models_{model_dir_name}')) for model_dir_name, _ in models)
            rows = math.ceil(math.sqrt(total_preds))
            cols = math.ceil(total_preds / rows)
            fig, axes = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True,
                                     figsize=(3 * cols, 3 * rows))

            axes = axes.flatten() if rows * cols > 1 else [axes]

            idx = 0
            for model_dir_name, model in models:
                total_params = sum(
                    param.numel() for param in model.parameters()
                )
                print(f'Total params of {model_dir_name}: {total_params}, memory needed = {total_params * (2 if next(model.parameters()).dtype == torch.float16 else 4)}')
                for version in os.listdir(f'models_{model_dir_name}'):
                    model_name = os.path.join(f'models_{model_dir_name}', version, 'best_val_loss.pth')
                    model.load_state_dict(torch.load(model_name, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                            start_event = torch.cuda.Event(enable_timing=True)
                            end_event = torch.cuda.Event(enable_timing=True)
                            start_event.record()
                        else:
                            start = time.time()

                        if model_dir_name == 'Unet_fp16':
                            pred = model(img_input.half())[0]
                        else:
                            pred = model(img_input)[0]

                        if device.type == "cuda":
                            end_event.record()
                            torch.cuda.synchronize()
                            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
                        else:
                            elapsed_time = time.time() - start

                    pred = pred.cpu().numpy().squeeze()
                    pred = (pred > 0.5).astype(np.float32)

                    overlaid_image = overlay_mask_on_image(img_np, pred)

                    axes[idx].imshow(overlaid_image, aspect='equal')
                    axes[idx].set_title(f'{"_".join(model_name.split("/")[:2])} in {elapsed_time:0.6f}s',
                                        fontsize=TITLE_FONT_SIZE, pad=20, wrap=True, color='blue')
                    axes[idx].axis('off')

                    individual_path = os.path.join(output_dir,
                                                   f'overlay_{model_dir_name}_{version}_{sub_dir}_{indice}.png')
                    cv2.imwrite(individual_path, cv2.cvtColor(overlaid_image, cv2.COLOR_RGB2BGR))

                    idx += 1

            for i in range(idx, len(axes)):
                axes[i].axis('off')

            plt.tight_layout(h_pad=12.0)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.97, bottom=0.02)
            output_path = os.path.join(output_dir, f'grid_output_{sub_dir}_{indice}.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig)


if __name__ == '__main__':
    TITLE_FONT_SIZE = 8
    IMG_HEIGHT, IMG_WIDTH = 256, 256
    DIR = '../car_pictures'
    SUB_DIRS = [
        '256_128',
        '320_180',
        '320_180_5deg',
        '320_180_10deg',
        '320_180_20deg',
        '320_180_outside',
        'UTAC_morning',
        'UTAC_morning2'
    ]
    NB_IMAGES_PER_DIRS = 10

    main()