import os
import math
import pickle
import random
import signal
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tabulate import tabulate
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import Pad
from tqdm import tqdm

from EfficientLiteSeg import EfficientLiteSeg
from Dataset import TrainingDataset, TestDataset
# from UNet_3Plus import UNet_3Plus
from evaluation import dice_coefficient, pixel_accuracy, iou_score


# from models.fast_scnn import FastSCNN
# from Unet import UNet
# from scripts.PolyRegression import PolyRegression


def display_results(model, dataset, device, train, sample_count):
    model.eval()

    indices = random.sample(range(len(dataset)), sample_count)
    for idx in indices:
        img = dataset[idx]
        if train:
            img, true_mask = dataset[idx]
        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            start = time.time()
            pred_mask = model(img_input)[0]
            print(f'did inference in {time.time() - start}')
        pred_mask_np = pred_mask.cpu().numpy().squeeze()
        pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)

        img_np = img.permute(1, 2, 0).cpu().numpy()
        if train:
            true_mask_np = true_mask.cpu().numpy().squeeze()

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Image")
        plt.axis("off")

        if train:
            plt.subplot(1, 3, 2)
            plt.imshow(true_mask_np, cmap="gray")
            plt.title("True Mask")
            plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask_np, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.show()


def save(path: str, values: [(str, [])], plot=False):
    torch.save(model.state_dict(), path + '.pth')
    print(f"Saved model checkpoint at: {path + '.pth'}")
    plot_training(values, plot=plot)


def main():
    transform_img = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        Pad((0, 6, 0, 6)),
        transforms.ToTensor(),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        Pad((0, 6, 0, 6)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(os.path.join(MODELS_DIR, VERSION)):
        os.makedirs(os.path.join(MODELS_DIR, VERSION))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4
    else:
        num_workers = 1

    if TRAIN:
        training(device, transform_img, transform_mask, num_workers)

    if TEST:
        testing(device, transform_img, transform_mask)


def testing(device, transform_img, transform_mask):
    TEST_IMAGES_DIR = "./../car_pictures/320_180"
    IMAGES_DIR = f"./../MaskGenerator/Dataset/{VERSION}/Images"
    MASKS_DIR = f"./../MaskGenerator/Dataset/{VERSION}/Masks"

    dataset = TrainingDataset(IMAGES_DIR, MASKS_DIR, transform_img=transform_img, transform_mask=transform_mask)
    dataset_test = TestDataset(TEST_IMAGES_DIR, transform_img=transform_img)

    # model = PolyRegression(num_outputs=1, backbone='resnet34', pretrained=False).to(device)
    # model = FastSCNN(num_classes=1).to(device)
    # model = UNet().to(device).half()
    model = EfficientLiteSeg(in_channels=3, out_channels=1).to(device)
    for param in model.parameters():
        print(param.dtype)
    # model = UNet_3Plus(in_channels=3, n_classes=1).to(device)
    print(f'Does model: {os.path.join(MODELS_DIR, VERSION, f"{MODEL_NAME}.pth")} exist ?')
    if os.path.isfile(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth')):
        print(f'Loading model: {os.path.join(MODELS_DIR, VERSION, f"{MODEL_NAME}.pth")}')
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth'), weights_only=True))
        print(f'Loaded model: {os.path.join(MODELS_DIR, VERSION, f"{MODEL_NAME}.pth")}')
    display_results(model=model, dataset=dataset, train=True, device=device, sample_count=5)
    display_results(model=model, dataset=dataset_test, train=False, device=device, sample_count=15)


def training(device, transform_img, transform_mask, num_workers):
    global model

    IMAGES_DIR = f"./../MaskGenerator/Dataset/{VERSION}/Images"
    MASKS_DIR = f"./../MaskGenerator/Dataset/{VERSION}/Masks"

    dataset = TrainingDataset(IMAGES_DIR, MASKS_DIR, transform_img=transform_img, transform_mask=transform_mask)

    generator = torch.Generator().manual_seed(25)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)],
                                              generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=num_workers, pin_memory=False,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    # model = PolyRegression(num_outputs=1, backbone='resnet34', pretrained=False).to(device)
    # model = FastSCNN(num_classes=1).to(device)
    # model = UNet().to(device).half()
    model = EfficientLiteSeg(in_channels=3, out_channels=1).to(device)
    # model = UNet_3Plus(in_channels=3, n_classes=1).to(device)

    if os.path.isfile(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth')):
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth'), weights_only=True))
        print(f'Loaded model: {os.path.join(MODELS_DIR, VERSION, f"{MODEL_NAME}.pth")}')

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(device=device)
    scaler = torch.amp.GradScaler('cuda')

    print(model)

    signal.signal(signal.SIGINT, handle_interrupt)

    train_losses = []
    train_dcs = []
    train_ious = []
    train_accs = []
    val_losses = []
    val_dcs = []
    val_ious = []
    val_accs = []

    nb_epoch_no_amelioration = 0
    last_val_loss = None
    best_val_loss_epoch = None
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0
        train_running_iou = 0
        train_running_acc = 0

        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.float16):
                y_pred = model(img)
                # y_pred = y_pred[0]

                dc = dice_coefficient(y_pred, mask)

                acc = pixel_accuracy(y_pred, mask)
                iou = iou_score(y_pred, mask)
                loss = criterion(y_pred, mask)
                # scale_factor = 65536.0  # Common scale for FP16
                # loss = loss * scale_factor
                # loss.backward()
                # for param in model.parameters():
                #     if param.grad is not None:
                #         param.grad.data /= scale_factor

                train_running_loss += loss.item()
                train_running_dc += dc.item()
                train_running_iou += iou
                train_running_acc += acc
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)
        train_iou = train_running_iou / (idx + 1)
        train_acc = train_running_acc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)
        train_ious.append(train_iou)
        train_accs.append(train_acc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0
        val_running_iou = 0
        val_running_acc = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    y_pred = model(img)
                    # y_pred = y_pred[0]
                    loss = criterion(y_pred, mask)
                    dc = dice_coefficient(y_pred, mask)

                    acc = pixel_accuracy(y_pred, mask)
                    iou = iou_score(y_pred, mask)

                val_running_loss += loss.item()
                val_running_dc += dc.item()
                val_running_iou += iou
                val_running_acc += acc

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)
            val_iou = val_running_iou / (idx + 1)
            val_acc = val_running_acc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)
        val_ious.append(val_iou)
        val_accs.append(val_acc)

        print("-" * 10 + f' EPOCH {epoch + 1} ' + "-" * 10)
        data = [
            ["", "train", "val"],
            ["Loss", train_loss, val_loss],
            ["DICE", train_dc, val_dc],
            ["Intersection Over Union", train_iou, val_iou],
            ["Pixel Accuracy", train_acc, val_acc],
        ]
        print(tabulate(data))
        print("-" * 30)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(MODELS_DIR, VERSION, f"model_epoch_{epoch + 1}")
            save(checkpoint_path, [
                ('train loss', train_losses),
                ('val loss', val_losses),
                ('train dice', train_dcs),
                ('val dice', val_dcs),
                ('train intersection over union', train_ious),
                ('val intersection over union', val_ious),
                ('train accuracy', train_accs),
                ('val accuracy', val_accs)
            ])

        if best_val_loss_epoch is None or best_val_loss_epoch > val_loss:
            checkpoint_path = os.path.join(MODELS_DIR, VERSION, f"best_val_loss")
            save(checkpoint_path, [
                ('train loss', train_losses),
                ('val loss', val_losses),
                ('train dice', train_dcs),
                ('val dice', val_dcs),
                ('train intersection over union', train_ious),
                ('val intersection over union', val_ious),
                ('train accuracy', train_accs),
                ('val accuracy', val_accs)
            ])
            best_val_loss_epoch = val_loss

        if last_val_loss is not None and last_val_loss <= val_loss:
            nb_epoch_no_amelioration += 1
        else:
            nb_epoch_no_amelioration = 0

        if nb_epoch_no_amelioration >= EARLY_STOPPING_PATIENCE or math.isnan(train_loss) or math.isnan(val_loss):
            print(f'Early stop after {epoch}')
            break

        last_val_loss = val_loss
    save(os.path.join(MODELS_DIR, VERSION, 'final_model'), [
        ('train loss', train_losses),
        ('val loss', val_losses),
        ('train dice', train_dcs),
        ('val dice', val_dcs),
        ('train intersection over union', train_ious),
        ('val intersection over union', val_ious),
        ('train accuracy', train_accs),
        ('val accuracy', val_accs)
    ], True)


def plot_training(values: [(str, [])], plot=False):
    fig = plt.figure()
    epochs_list = list(range(1, len(values[0][1]) + 1))
    for i in range(len(values)):
        lines = []
        line, = plt.plot(epochs_list, values[i][1], label=values[i][0])
        lines.append(line)

    plt.grid()
    plt.legend()

    plt.savefig(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}_plot.png'), dpi=300, bbox_inches='tight')
    with open(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}_plot.plot'), 'wb') as f:
        pickle.dump(fig, f)
    if plot:
        plt.show()
    plt.close()


def toggle_visibility(label, lines, labels):
    index = labels.index(label)
    lines[index].set_visible(not lines[index].get_visible())
    plt.draw()


def handle_interrupt(signal, frame):
    print("\nTraining interrupted by user. Saving model as final_model...")
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, VERSION, 'final_model.pth'))
    print("Model saved successfully.")
    sys.exit(0)


if __name__ == '__main__':
    IMG_HEIGHT, IMG_WIDTH = 180, 320
    BATCH_SIZE = 32
    EPOCHS = 2
    LEARNING_RATE = 1e-5
    EARLY_STOPPING_PATIENCE = 4
    VERSION = '0.20'
    MODEL_NAME = 'final_model'
    MODELS_DIR = 'models_Unet_fp16'
    TRAIN = False
    TEST = True
    eps = 1e-5
    ignore = True
    average = True

    torch.cuda.empty_cache()
    main()
    torch.cuda.empty_cache()
