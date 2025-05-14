import os
import random
import time

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import Pad

from Dataset import TestDataset
from UNet_3Plus import UNet_3Plus
from models.fast_scnn import FastSCNN
from Unet import UNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f'Using device: {device}')

    models = [
        ('fastSCNN', FastSCNN(num_classes=1).to(device)),
        ('UNet_3Plus', UNet_3Plus(in_channels=3, n_classes=1).to(device)),
        ('Unet', UNet(in_channels=3, out_channels=1).to(device)),
    ]

    transform_img = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        Pad((0, 6, 0, 6)),
        transforms.ToTensor(),
    ])

    for sub_dir in SUB_DIRS:
        dataset_test = TestDataset(os.path.join(DIR, sub_dir), transform_img=transform_img)
        indices = random.sample(range(len(dataset_test)), NB_IMAGES_PER_DIRS)

        for indice in indices:
            image = dataset_test[indice]
            img_input = image.unsqueeze(0).to(device)
            fig, axes = plt.subplots(nrows=4, ncols=(len(models) + 1) // 4 + 1, sharex=True, sharey=True)

            idx = 0
            for model_dir_name, model in models:
                for version in os.listdir(f'models_{model_dir_name}'):
                    model_name = os.path.join(f'models_{model_dir_name}', version, 'final_model.pth')
                    model.load_state_dict(torch.load(model_name, map_location=device))
                    model.eval()
                    with torch.no_grad():
                        start = time.time()
                        pred = model(img_input)[0]
                        end = time.time()
                    pred = pred.cpu().numpy().squeeze()
                    pred = (pred > 0.5).astype(np.float32)
                    axes.flat[idx].imshow(pred, cmap='gray')
                    axes.flat[idx].set_title(f'{model_name} in {end - start:0.6f}', fontsize=TITLE_FONT_SIZE)
                    axes.flat[idx].axis('off')
                    idx += 1
            img_np = image.permute(1, 2, 0).cpu().numpy()
            axes.flat[idx].imshow(img_np)
            axes.flat[idx].set_title(f'Original', fontsize=TITLE_FONT_SIZE)
            axes.flat[idx].axis('off')
            plt.show()

if __name__ == '__main__':
    TITLE_FONT_SIZE = 5
    IMG_HEIGHT, IMG_WIDTH = 180, 320
    DIR = '../car_pictures'
    SUB_DIRS = ['320_180', '256_128']
    NB_IMAGES_PER_DIRS = 10

    main()