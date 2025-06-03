import math
import os
import re
import traceback

import cv2
import pandas as pd
import random
from collections import defaultdict
import numpy as np


def min_max_normalize(arr, new_min=0, new_max=1):
    arr_min = np.min(arr[(arr > 0)])
    arr_max = np.max(arr)
    # Avoid division by zero
    if arr_max == arr_min:
        return np.zeros_like(arr)
    normalized = (arr - arr_min) / (arr_max - arr_min) * (new_max - new_min) + new_min
    return normalized


# def compute_pred(raycast, nb_group=5):
#     # print(raycast)
#     groups = []
#     raycast[0] = min_max_normalize(raycast[0])
#     # print(raycast)
#
#     for i in range(0, len(raycast[0]), len(raycast[0]) // nb_group):
#         group = raycast[0][i:i + len(raycast[0]) // nb_group]
#         group = np.where(group < 0, 1.5, group)
#         groups.append(group)
#     # print(groups)
#     averages = []
#     for group in groups:
#         averages.append(np.average(group))
#
#     diffs = []
#     # print(averages)
#     for i in range(len(averages) // 2):
#         diffs.append(averages[i] - averages[-i - 1])
#     # print(diffs)
#
#     # print(sum(diffs) / len(diffs))
#     return 1 - (max(min(sum(diffs) / len(diffs) * (1 - averages[len(averages) // 2]), 1.0), -1.0) + 1) / 2
def compute_pred(raycast, nb_group=5):
    print(raycast)
    groups = []
    raycast = min_max_normalize(raycast)
    # print(raycast)

    for i in range(0, len(raycast), len(raycast) // nb_group):
        group = raycast[i:i + len(raycast) // nb_group]
        group = np.where(group < 0, 1.4, group)
        groups.append(group)
    # print(groups)
    averages = []
    for group in groups:
        averages.append(np.average(group))

    diffs = []
    # print(averages)
    for i in range(len(averages) // 2):
        diffs.append(averages[i] - averages[-i - 1])
    # print(diffs)

    #steer = 1 - (max(min(sum(diffs) / len(diffs) * (1 - averages[len(averages) // 2]), 1.0), -1.0) + 1) / 2
    steer = sum(diffs) / len(diffs)

    central_line_dist = averages[len(averages) // 2]
    # last_10_average = sum(last_10_decisions) / len(last_10_decisions)
    last_10_average = 0.3
    print(f'steer without: {1 - (max(min(steer, 1.0), -1.0) + 1) / 2}')
    steer -= (steer - last_10_average) * (1 - central_line_dist) / 2

    steer = 1 - (max(min(steer, 1.0), -1.0) + 1) / 2

    # last_10_decisions.append(steer)
    # print(len(last_10_decisions))


    # print(sum(diffs) / len(diffs))
    return steer


def get_numeric_prefix(filename):
    match = re.match(r"(\d+)_", filename)
    return int(match.group(1)) if match else None


def display_random_grouped_images(folder_path, num_groups_to_display=3):
    # Find CSV file in the folder
    csv_file = None
    for fname in os.listdir(folder_path):
        if fname.lower().endswith('.csv'):
            csv_file = os.path.join(folder_path, fname)
            break

    if not csv_file:
        print("No CSV file found in the folder.")
        return

    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Failed to read CSV file: {e}")
        return

    # Group images by numeric prefix
    groups = defaultdict(list)
    for fname in sorted(os.listdir(folder_path)):
        prefix = get_numeric_prefix(fname)
        if prefix is not None and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            groups[prefix].append(fname)

    if not groups:
        print("No valid images found in the folder.")
        return

    # Ensure we don't request more groups than available
    available_prefixes = list(groups.keys())
    num_to_pick = min(num_groups_to_display, len(available_prefixes))
    selected_prefixes = random.sample(available_prefixes, num_to_pick)

    for random_prefix in selected_prefixes:
        print(f"\nShowing group: {random_prefix}")
        # Load images for the random prefix
        images = []
        for fname in sorted(groups[random_prefix]):
            full_path = os.path.join(folder_path, fname)
            img = cv2.imread(full_path)
            if img is not None:
                img = cv2.resize(img, (320, 240))
                images.append(img)
            else:
                print(f"Failed to load image: {fname}")

        # Get corresponding CSV row by line number (index)
        text_to_display = ""
        try:
            if 0 <= random_prefix + 2 < len(df):  # Ensure within bounds
                decision = df.iloc[random_prefix + 2]['decision']
                direction = df.iloc[random_prefix + 2]['direction']

                raycast_columns = [col for col in df.columns if col.startswith("ray_cast_")]
                raycasts = df.loc[random_prefix + 2, raycast_columns].values.reshape(1, -1)

                prediction = compute_pred(raycasts[0])
                print("Predicted value:", prediction)

                text_to_display = f"Direction: {direction:.4f}, Prediction: {prediction:.4f}"
            else:
                text_to_display = f"No CSV row at index {random_prefix + 2}"
                print(text_to_display)
        except Exception as e:
            text_to_display = f"Error accessing CSV data: {e}"
            print(text_to_display)
            traceback.print_exc()

        # Display images with text
        if images:
            combined = np.hstack(images)
            text_height = 50
            combined_with_text = np.zeros((240 + text_height, combined.shape[1], 3), dtype=np.uint8)
            combined_with_text[:] = (255, 255, 255)
            combined_with_text[text_height:, :, :] = combined

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_color = (0, 0, 0)
            thickness = 2
            text_position = (10, 30)
            cv2.putText(combined_with_text, text_to_display, text_position, font, font_scale, font_color, thickness)

            cv2.imshow(f"Group {random_prefix}", combined_with_text)
            cv2.waitKey(0)
            cv2.destroyWindow(f"Group {random_prefix}")
        else:
            print(f"No images to display for group {random_prefix}.")

    cv2.destroyAllWindows()


# Example usage
folder_path = "../../../Downloads/dataset_20250520_145848"  # Change to your folder path
display_random_grouped_images(folder_path, num_groups_to_display=20)
