import glob
import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm


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
    # Two consecutive convolutional layers with ReLU activation and same padding.
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
            pred_mask = model(img_input)[0]
        # Convert tensor to numpy array and threshold the prediction
        pred_mask_np = pred_mask.cpu().numpy().squeeze()
        pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)

        # Convert image and true mask to numpy arrays for display
        img_np = img.permute(1, 2, 0).cpu().numpy()
        if train:
            true_mask_np = true_mask.cpu().numpy().squeeze()

        plt.figure(figsize=(12, 4))
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


def main():
    train = False
    test = True

    transform_img = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    if not os.path.exists("models"):
        os.makedirs("models")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        num_workers = torch.cuda.device_count() * 4
    else:
        num_workers = 1

    if train:
        training(device, transform_img, transform_mask, num_workers)

    if test:
        testing(device, transform_img)


def testing(device, transform_img):
    TEST_IMAGES_DIR = "./../carpet/output_images_carpet"
    MODEL_NAME = 'best_val_loss'

    dataset_test = TestDataset(TEST_IMAGES_DIR, transform_img=transform_img)

    model = UNet(in_channels=3, out_channels=1).to(device)
    if os.path.isfile(os.path.join('models', f'{MODEL_NAME}.pth')):
        model.load_state_dict(torch.load(os.path.join('models', f'{MODEL_NAME}.pth'), weights_only=True))
    display_results(model=model, dataset=dataset_test, train=False, device=device, sample_count=15)


def training(device, transform_img, transform_mask, num_workers):
    version = '0.17'
    IMAGES_DIR = f"./../MaskGenerator/Dataset/{version}/Images"
    MASKS_DIR = f"./../MaskGenerator/Dataset/{version}/Masks"

    dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR, transform_img=transform_img, transform_mask=transform_mask)

    generator = torch.Generator().manual_seed(25)
    train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  num_workers=num_workers, pin_memory=False,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset,
                                num_workers=num_workers, pin_memory=False,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(device=device)
    scaler = torch.amp.GradScaler('cuda')

    print(model)

    train_losses = []
    train_dcs = []
    val_losses = []
    val_dcs = []

    nb_epoch_no_amelioration = 0
    last_val_loss = None
    best_val_loss_epoch = None
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_dc = 0

        for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                y_pred = model(img)

                dc = dice_coefficient(y_pred, mask)
                loss = criterion(y_pred, mask)

            train_running_loss += loss.item()
            train_running_dc += dc.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        train_loss = train_running_loss / (idx + 1)
        train_dc = train_running_dc / (idx + 1)

        train_losses.append(train_loss)
        train_dcs.append(train_dc)

        model.eval()
        val_running_loss = 0
        val_running_dc = 0

        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader, position=0, leave=True)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                with torch.amp.autocast('cuda'):
                    y_pred = model(img)
                    loss = criterion(y_pred, mask)
                    dc = dice_coefficient(y_pred, mask)

                val_running_loss += loss.item()
                val_running_dc += dc.item()

            val_loss = val_running_loss / (idx + 1)
            val_dc = val_running_dc / (idx + 1)

        val_losses.append(val_loss)
        val_dcs.append(val_dc)

        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
        print("-" * 30)

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join("models", f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint at: {checkpoint_path}")

        if best_val_loss_epoch is None or best_val_loss_epoch > val_loss:
            checkpoint_path = os.path.join("models", f"best_val_loss.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint at: {checkpoint_path}")
            best_val_loss_epoch = val_loss

        if last_val_loss is not None and last_val_loss <= val_loss:
            nb_epoch_no_amelioration += 1
        else:
            nb_epoch_no_amelioration = 0

        if nb_epoch_no_amelioration >= EARLY_STOPPING_PATIENCE or math.isnan(train_loss) or math.isnan(val_loss):
            print(f'Early stop after {epoch}')
            break

        last_val_loss = val_loss
    torch.save(model.state_dict(), os.path.join('models', 'final_model.pth'))
    epochs_list = list(range(1, len(train_losses) + 1))
    plot_training(epochs_list, train_losses, val_losses, train_dcs, val_dcs)
    display_results(model=model, dataset=dataset, train=True, device=device, sample_count=3)

def plot_training(epochs_list, train_losses, val_losses, train_dcs, val_dcs):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, train_losses, label='Training Loss')
    plt.plot(epochs_list, val_losses, label='Validation Loss')
    plt.xticks(ticks=list(range(1, EPOCHS + 1, 1)))
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.grid()
    plt.tight_layout()

    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, train_dcs, label='Training DICE')
    plt.plot(epochs_list, val_dcs, label='Validation DICE')
    plt.xticks(ticks=list(range(1, EPOCHS + 1, 1)))
    plt.title('DICE Coefficient over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('DICE')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    IMG_HEIGHT, IMG_WIDTH = 128, 256
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 5

    torch.cuda.empty_cache()
    main()
    torch.cuda.empty_cache()
