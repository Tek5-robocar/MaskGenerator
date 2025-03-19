import os
import re
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torch import optim, nn
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image
import gc
gc.collect()
torch.cuda.empty_cache()


WORKING_DIR = os.getcwd()
image_directory = WORKING_DIR + "/Dataset/Images"
mask_directory = WORKING_DIR + "/Dataset/Masks"
SIZE = (128,256)
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_CLASS = 2
EPOCHS = 0

if (os.path.exists(WORKING_DIR + 'checkpoint') == False):
    os.mkdir(WORKING_DIR + 'checkpoint')
if (os.path.exists(WORKING_DIR + 'final') == False):
    os.mkdir(WORKING_DIR + 'final')

class SimDataset(Dataset):
  def __init__(self, image_paths, mask_paths_left, mask_paths_right, transform_img=None, transform_mask=None):
    self.image_paths = image_paths
    self.mask_paths_left = mask_paths_left
    self.mask_paths_right = mask_paths_right
    self.transform_img = transform_img
    self.transform_mask = transform_mask

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    mask_path_left = self.mask_paths_left[idx]
    mask_path_right = self.mask_paths_right[idx]

    mask_left = Image.open(mask_path_left).convert("L") # convert("L") will convert the image to grayscale
    mask_right = Image.open(mask_path_right).convert("L") # convert("L") will convert the image to grayscale
    if self.transform_mask:
        mask_left = self.transform_mask(mask_left)
        mask_right = self.transform_mask(mask_right)
        mask = torch.tensor(np.array([mask_left[0], mask_right[0]]))
        mask[mask > 0.0] = 1

    image = Image.open(img_path).convert("RGB") # we might not need to do this because the images are loaded as RGB by default
    if self.transform_img:
      image = self.transform_img(image)
    return [image, mask]

image_paths = sorted(glob.glob(os.path.join(image_directory, "*.png")), key=lambda x:float(re.findall("(\d+)",x)[-1]))
mask_paths_left = sorted(glob.glob(os.path.join(mask_directory, "*-left.png")), key=lambda x:float(re.findall("(\d+)",x)[-1]))
mask_paths_right = sorted(glob.glob(os.path.join(mask_directory, "*-right.png")), key=lambda x:float(re.findall("(\d+)",x)[-1]))
assert len(mask_paths_left) == len(mask_paths_right), f"Amount of left and right mask is different {len(mask_paths_left)} != {len(mask_paths_right)}"
assert len(image_paths) == len(mask_paths_left), f"Amount of images and label are different {len(image_paths)} != {len(mask_paths_left)}"
dataset = SimDataset(image_paths,
                     mask_paths_left,
                     mask_paths_right,
                     transforms.Compose([
                                         transforms.Resize(SIZE),
                                         transforms.ColorJitter(brightness=.5, hue=.3),
                                         transforms.ToTensor()
                                         #transforms.ToImage(),
                                         #transforms.ToDtype(torch.float32, scale=True),
                                         # transforms.Normalize(mean=[0.0],
                                         #                      std=[1.0])
                                                              ]),
                     transforms.Compose([
                                         transforms.Resize(SIZE),
                                         transforms.ToTensor()
                                         #transforms.ToImage(),
                                         #transforms.ToDtype(torch.float32, scale=True),
                                         # transforms.Normalize(mean=[0.0],
                                         #                       std=[1.0])
                                                              ]))
generator = torch.Generator().manual_seed(25)
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
test_dataset, val_dataset = random_split(test_dataset, [0.25, 0.75], generator=generator)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)

        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], 1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)

        out = self.out(up_4)
        return out


torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    num_workers = torch.cuda.device_count() * 4
else:
    num_workers = 1

train_dataloader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

val_dataloader = DataLoader(dataset=val_dataset,
                            num_workers=num_workers, pin_memory=False,
                            batch_size=BATCH_SIZE,
                            shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset,
                             num_workers=num_workers, pin_memory=False,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

# model_pth = '/kaggle/working/final/final_epoch30.pth'

model = UNet(in_channels=3, num_classes=NUM_CLASS).to(device)
# model.load_state_dict(torch.load(model_pth,weights_only=True, map_location=torch.device(device)))
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss().to(device=device)
scaler = torch.amp.GradScaler('cuda')

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.float().clone()
    target = target.float()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice


torch.cuda.empty_cache()

train_losses = []
train_dcs = []
val_losses = []
val_dcs = []

checkpoint_dir = "models"
os.makedirs(checkpoint_dir, exist_ok=True)  

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
    print("\n")
    print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
    print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
    print("-" * 30)
    if (epoch > 9 and epoch % 10 == 0):
        torch.save(model.state_dict(), f"models/checkpoint_epoch{epoch}.pth")

# Saving the model
print("save")
torch.save(model.state_dict(), f'models/final_epoch{EPOCHS}_fp16.pth')

epochs_list = list(range(1, EPOCHS + 1))

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

model_pth = WORKING_DIR + '/models/final_epoch30_fp16.pth'
trained_model = UNet(in_channels=3, num_classes=NUM_CLASS).half().to(device)
trained_model.load_state_dict(torch.load(model_pth,weights_only=True, map_location=torch.device(device)))

for name, param in trained_model.named_parameters():
    print(f"{name}: {param.dtype}")

test_running_loss = 0
test_running_dc = 0

with torch.no_grad():
    for idx, img_mask in enumerate(tqdm(test_dataloader, position=0, leave=True)):
        img = img_mask[0].float().half().to(device)
        mask = img_mask[1].float().half().to(device)

        y_pred = trained_model(img)
        loss = criterion(y_pred, mask)
        dc = dice_coefficient(y_pred, mask)

        test_running_loss += loss.item()
        test_running_dc += dc.item()

    test_loss = test_running_loss / (idx + 1)
    test_dc = test_running_dc / (idx + 1)
print(f"{test_loss=}")
print(f"{test_dc=}")

def random_images_inference(image_tensors, mask_tensors, model_pth, device):
    model = UNet(in_channels=3, num_classes=NUM_CLASS).half().to(device)
    model.load_state_dict(torch.load(model_pth, weights_only=True, map_location=torch.device(device)))

    transform = transforms.Compose([
        transforms.Resize(SIZE)
    ])

    # Iterate for the images, masks and paths
    for image_pth, mask_pth in zip(image_tensors, mask_tensors):
        # Load the image
        #img = image_pth
        img = transform(image_pth)
        
        # Predict the imagen with the model
        pred_mask = model(img.unsqueeze(0).float().half().to(device))
        pred_mask = pred_mask.squeeze(0) 
        pred_mask = pred_mask.permute(1,2,0)
        
        # Load the mask to compare
#        mask = transform(mask_pth).permute(1, 2, 0).to(device)
        mask = mask_pth.permute(1, 2, 0).cpu().detach()
        mask = mask.permute(2, 0, 1)
        merged_mask = mask[0] + mask[1]
        
        # Show the images
        img = img.cpu().detach().permute(1, 2, 0)
        pred_mask = pred_mask.cpu().detach()
        pred_mask = pred_mask.permute(2,0,1)
        pred_mask[0] = torch.sigmoid(pred_mask[0]) > 0.5
        pred_mask[1] = torch.sigmoid(pred_mask[1]) > 0.5
        merged_pred_mask = pred_mask[0] + pred_mask[1]

    
        plt.figure(figsize=(16, 4))
        plt.subplot(2,4,1), plt.imshow(img), plt.title("original"), plt.axis("off")
        plt.subplot(2,4,2), plt.imshow(pred_mask[0], cmap="gray"), plt.title("predicted left"), plt.axis("off")
        plt.subplot(2,4,3), plt.imshow(pred_mask[1], cmap="gray"), plt.title("predicted right"), plt.axis("off")
        plt.subplot(2,4,4), plt.imshow(merged_pred_mask, cmap="gray"), plt.title("predicted merged"), plt.axis("off")
        
        plt.subplot(2,4,6), plt.imshow(mask[0], cmap="gray"), plt.title("mask left"), plt.axis("off")
        plt.subplot(2,4,7), plt.imshow(mask[1], cmap="gray"), plt.title("mask right"), plt.axis("off")
        plt.subplot(2,4,8), plt.imshow(merged_mask, cmap="gray"), plt.title("mask merged"), plt.axis("off")
        plt.show()

n = 20

image_tensors = []
mask_tensors = []
image_paths = []

for _ in range(n):
    random_index = random.randint(0, len(test_dataloader.dataset) - 1)
    random_sample = test_dataloader.dataset[random_index]

    image_tensors.append(random_sample[0])  
    mask_tensors.append(random_sample[1])

model_path =  WORKING_DIR + '/models/final_epoch30_fp16.pth'

random_images_inference(image_tensors, mask_tensors, model_path, device="cuda")




#resize nearest neighbor pour les mask