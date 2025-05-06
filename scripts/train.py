import glob
import math
import os
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
from PIL import Image
from tabulate import tabulate
from torch.nn.functional import pad
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# from scripts.UNet_3Plus import UNet_3Plus


# Custom dataset for paired images and masks
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform_img=None, transform_mask=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (RGB) and mask (grayscale)
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize images and masks
        if self.transform_img is not None:
            img = self.transform_img(img)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        else:
            # Default conversion: ToTensor scales images to [0,1]
            mask = transforms.ToTensor()(mask)

        # Ensure mask is binary (values 0 or 1)
        mask = (mask > 0.5).float()
        return img, mask


class TestDataset(Dataset):
    def __init__(self, images_dir, transform_img=None):
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.png")))
        self.transform_img = transform_img

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image (RGB) and mask (grayscale)
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Resize images and masks
        if self.transform_img is not None:
            img = self.transform_img(img)

        # Ensure mask is binary (values 0 or 1)
        return img


# U-Net Model in PyTorch
def double_conv(in_channels, out_channels):
    #     Two consecutive convolutional layers with ReLU activation and same padding.
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = double_conv(512, 1024)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = double_conv(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)

        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        if x.size()[2] == 180:
            x = pad(x, (0, 0, 6, 6))

        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)
        p3 = self.pool3(c3)

        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder
        u4 = self.upconv4(bn)
        # Concatenate u4 and c4 along the channel dimension
        u4 = torch.cat([u4, c4], dim=1)
        d4 = self.dec4(u4)

        u3 = self.upconv3(d4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.upconv2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.upconv1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)

        out = self.conv_last(d1)
        return out


# After training, display sample predictions.
def display_results(model, dataset, device, train, sample_count):
    model.eval()

    indices = random.sample(range(len(dataset)), sample_count)
    for idx in indices:
        img = dataset[idx]
        if train:
            img, true_mask = dataset[idx]
        # Add batch dimension and move to device
        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            start = time.time()
            pred_mask = model(img_input)[0]
            print(f'did inference in {time.time() - start}')
        # Convert tensor to numpy array and threshold the prediction
        pred_mask_np = pred_mask.cpu().numpy().squeeze()
        pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)

        # Convert image and true mask to numpy arrays for display
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


def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.float().clone()
    target = target.float()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice


def save(path: str, values: [(str, [])], plot=False):
    torch.save(model.state_dict(), path + '.pth')
    print(f"Saved model checkpoint at: {path + '.pth'}")
    plot_training(values, plot=plot)


def main():
    transform_img = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
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

    dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR, transform_img=transform_img, transform_mask=transform_mask)
    dataset_test = TestDataset(TEST_IMAGES_DIR, transform_img=transform_img)

    model = UNet(in_channels=3, out_channels=1).to(device)
    if os.path.isfile(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth')):
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth'), weights_only=True))
    display_results(model=model, dataset=dataset, train=True, device=device, sample_count=5)
    display_results(model=model, dataset=dataset_test, train=False, device=device, sample_count=15)


def pixel_accuracy(prediction, target):
    """
    Computes pixel accuracy between prediction and target masks.

    Args:
        prediction (torch.Tensor): Predicted mask (logits or probabilities).
        target (torch.Tensor): Ground truth mask (binary or multi-class).

    Returns:
        float: Pixel accuracy in [0, 1].
    """
    # Convert to binary if needed (assuming binary segmentation)
    if prediction.dtype != torch.bool:
        pred_labels = (prediction > 0).float()  # threshold at 0 for logits
    else:
        pred_labels = prediction.float()

    target_labels = target.float()

    correct_pixels = torch.sum(pred_labels == target_labels)
    total_pixels = torch.numel(target_labels)

    accuracy = correct_pixels / total_pixels

    return accuracy.item()  # return as Python float


def iou_score(prediction, target, epsilon=1e-7):
    """
    Computes Intersection over Union (IoU) between prediction and target masks.

    Args:
        prediction (torch.Tensor): Predicted mask (logits or probabilities).
        target (torch.Tensor): Ground truth mask (binary or multi-class).
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        float: IoU score in [0, 1].
    """
    # Binarize predictions (assuming binary segmentation)
    pred_labels = (prediction > 0).float()
    target_labels = target.float()

    intersection = torch.sum(pred_labels * target_labels)
    union = torch.sum(pred_labels) + torch.sum(target_labels) - intersection

    iou = (intersection + epsilon) / (union + epsilon)

    return iou.item()


def training(device, transform_img, transform_mask, num_workers):
    global model

    IMAGES_DIR = f"./../MaskGenerator/Dataset/{VERSION}/Images"
    MASKS_DIR = f"./../MaskGenerator/Dataset/{VERSION}/Masks"

    dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR, transform_img=transform_img, transform_mask=transform_mask)

    generator = torch.Generator().manual_seed(25)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)], generator=generator)


    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=num_workers, pin_memory=False,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)

    if os.path.isfile(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth')):
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, VERSION, f'{MODEL_NAME}.pth'), weights_only=True))

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(device=device)
    scaler = torch.amp.GradScaler('cuda')

    print(model)

    signal.signal(signal.SIGINT, handle_interrupt)

    train_losses = []
    train_dcs = []
    # train_dcs1 = []
    train_ious = []
    train_accs = []
    val_losses = []
    val_dcs = []
    # val_dcs1 = []
    val_ious = []
    val_accs = []

    nb_epoch_no_amelioration = 0
    last_val_loss = None
    best_val_loss_epoch = None
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0
        # train_running_dc1 = 0
        train_running_iou = 0
        train_running_acc = 0

        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            if img.size()[2] == 180:
                img = pad(img, (0, 0, 6, 6))

            if mask.size()[2] == 180:
                mask = pad(mask, (0, 0, 6, 6))

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                y_pred = model(img)

                dc = dice_coefficient(y_pred, mask)

                # y_true_flat = mask.cpu().detach().numpy().flatten()
                # y_true_flat = np.nan_to_num(y_true_flat)
                # y_true_flat = np.clip(y_true_flat, 0, 255).astype(np.uint8)
                # y_pred_flat = y_pred.cpu().detach().numpy().flatten()
                # y_pred_flat = np.nan_to_num(y_pred_flat)
                # y_pred_flat = np.clip(y_pred_flat, 0, 255).astype(np.uint8)
                acc = pixel_accuracy(y_pred, mask)
                iou = iou_score(y_pred, mask)
                # dc1 = f1_score(y_true_flat, y_pred_flat)
                # iou = jaccard_score(y_true_flat, y_pred_flat)
                # acc = np.mean(y_true_flat == y_pred_flat)
                loss = criterion(y_pred, mask)

            train_running_loss += loss.item()
            train_running_dc += dc.item()
            # train_running_dc1 += dc1
            train_running_iou += iou
            train_running_acc += acc
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)
        # train_dc1 = train_running_dc1 / (idx + 1)
        train_iou = train_running_iou / (idx + 1)
        train_acc = train_running_acc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)
        # train_dcs1.append(train_dc1)
        train_ious.append(train_iou)
        train_accs.append(train_acc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0
        # val_running_dc1 = 0
        val_running_iou = 0
        val_running_acc = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                if img.size()[2] == 180:
                    img = pad(img, (0, 0, 6, 6))

                if mask.size()[2] == 180:
                    mask = pad(mask, (0, 0, 6, 6))

                with torch.amp.autocast('cuda'):
                    y_pred = model(img)
                    loss = criterion(y_pred, mask)
                    dc = dice_coefficient(y_pred, mask)

                    # y_true_flat = mask.cpu().detach().numpy().flatten().astype(np.uint8)
                    # y_pred_flat = y_pred.cpu().detach().numpy().flatten().astype(np.uint8)
                    class_num = y_pred.size(1)
                    acc = pixel_accuracy(y_pred, mask)
                    iou = iou_score(y_pred, mask)
                    # dc1 = f1_score(y_true_flat, y_pred_flat)
                    # iou = jaccard_score(y_true_flat, y_pred_flat)
                    # acc = np.mean(y_true_flat == y_pred_flat)

                val_running_loss += loss.item()
                val_running_dc += dc.item()
                # val_running_dc1 += dc1
                val_running_iou += iou
                val_running_acc += acc

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)
            # val_dc1 = val_running_dc1 / (idx + 1)
            val_iou = val_running_iou / (idx + 1)
            val_acc = val_running_acc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)
        # val_dcs1.append(val_dc1)
        val_ious.append(val_iou)
        val_accs.append(val_acc)

        print("-" * 10 + f' EPOCH {epoch + 1} ' + "-" * 10)
        data = [
            ["", "train", "val"],
            ["Loss", train_loss, val_loss],
            ["DICE", train_dc, val_dc],
            # ["DICE 1", train_dc1, val_dc1],
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
                # ('train dice 1', train_dcs1),
                # ('va dice 1', val_dcs1),
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
                # ('train dice 1', train_dcs1),
                # ('va dice 1', val_dcs1),
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
        # ('train dice 1', train_dcs1),
        # ('va dice 1', val_dcs1),
        ('train intersection over union', train_ious),
        ('val intersection over union', val_ious),
        ('train accuracy', train_accs),
        ('val accuracy', val_accs)
    ], True)


def plot_training(values: [(str, [])], plot=False):
    fig = plt.figure()
    epochs_list = list(range(1, len(values[0][1]) + 1))
    for i in range(len(values)):
        # lines = []
        line, = plt.plot(epochs_list, values[i][1], label=values[i][0])

    #     lines.append(line)

    # plt.grid()
    # plt.legend()

    # rax = plt.axes([0.02, 0.4, 0.15, 0.15])

    # labels = [line.get_label() for line in lines]
    # visibility = [line.get_visible() for line in lines]
    # check = CheckButtons(rax, labels, visibility)

    # check.on_clicked(lambda label: toggle_visibility(label, lines, labels))
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
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 4
    VERSION = '0.20'
    MODEL_NAME = 'final_model'
    MODELS_DIR = 'old_models'
    TRAIN = False
    TEST = True
    eps = 1e-5
    ignore = True
    average = True

    import gc
    import gc

    # del variables
    # gc.collect()
    torch.cuda.empty_cache()
    main()
    torch.cuda.empty_cache()
