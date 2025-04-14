import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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
        self.enc1 = double_conv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = double_conv(16, 32)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = double_conv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = double_conv(64, 128)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = double_conv(128, 256)

        self.upconv4 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec4 = double_conv(256, 128)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = double_conv(128, 64)

        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = double_conv(64, 32)

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = double_conv(32, 16)

        self.conv_last = nn.Conv2d(16, out_channels, kernel_size=1)

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
        # Sigmoid activation for binary segmentation
        return torch.sigmoid(out)


# After training, display sample predictions.
def display_results(model, dataset, device, sample_count=3):
    model.eval()
    indices = random.sample(range(len(dataset)), sample_count)
    for idx in indices:
        img = dataset[idx]
        # img, true_mask = dataset[idx]
        # Add batch dimension and move to device
        img_input = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred_mask = model(img_input)[0]
        # Convert tensor to numpy array and threshold the prediction
        pred_mask_np = pred_mask.cpu().numpy().squeeze()
        pred_mask_np = (pred_mask_np > 0.5).astype(np.float32)

        # Convert image and true mask to numpy arrays for display
        img_np = img.permute(1, 2, 0).cpu().numpy()
        # true_mask_np = true_mask.cpu().numpy().squeeze()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Image")
        plt.axis("off")

        # plt.subplot(1, 3, 2)
        # plt.imshow(true_mask_np, cmap="gray")
        # plt.title("True Mask")
        # plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask_np, cmap="gray")
        plt.title("Predicted Mask")
        plt.axis("off")
        plt.show()


if __name__ == '__main__':
    train = False

    # Parameters
    IMG_HEIGHT, IMG_WIDTH = 256, 256
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Directories for images and masks; adjust as needed
    TEST_IMAGES_DIR = "./../carpet/output_images_carpet"

    if train:
        IMAGES_DIR = "./../MaskGenerator/Dataset/Images"
        MASKS_DIR = "./../MaskGenerator/Dataset/Masks"

    # Create directory for saving models if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")

    # Define image and mask transformations
    transform_img = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),  # converts to tensor and scales image to [0,1]
    ])

    transform_mask = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),  # converts to tensor; assumes mask is grayscale
    ])

    # Create dataset and data loader
    dataset = TestDataset(TEST_IMAGES_DIR, transform_img=transform_img)
    if train:
        dataset = SegmentationDataset(IMAGES_DIR, MASKS_DIR, transform_img=transform_img, transform_mask=transform_mask)
    # Optional: split into train and validation (here we use the full dataset for training)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Instantiate model, loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss()  # binary cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if os.path.isfile(os.path.join('models', 'model_epoch_50.pth')):
        model.load_state_dict(torch.load(os.path.join('models', 'model_epoch_50.pth'), weights_only=True))

    print(model)

    # Training loop with model checkpoint saving every 10 epochs
    if train:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            for imgs, masks in dataloader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * imgs.size(0)

            epoch_loss /= len(dataset)
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}")

            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join("models", f"model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved model checkpoint at: {checkpoint_path}")

    display_results(model, dataset, device)
