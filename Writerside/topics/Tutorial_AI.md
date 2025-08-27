# Tutorial

This tutorial will guide you through training a **UNet** model on a dataset generated from a Unity simulation. It assumes you have some basic knowledge of Python and PyTorch.

You will learn how to:

* Load an existing dataset of images and masks
* Preprocess and prepare the data
* Train a UNet for image segmentation
* Evaluate the model

---

## Before You Start

Make sure you have:

- Python installed (>= 3.8)
- Required libraries installed:

```bash
pip install -r requirements.txt
```

- Access to the dataset of images and masks generated from the Unity simulation
- Basic knowledge of PyTorch and neural networks

---

## Part 1: Loading the Dataset

The dataset is assumed to be already generated and stored in folders, for example:

- `dataset/images/` → input images
- `dataset/masks/` → corresponding segmentation masks

You need to **load the dataset using PyTorch's `DataLoader`**, applying any necessary transforms such as normalization or conversion to tensors.

---

## Part 2: Creating the UNet Model

For the segmentation model, we'll use **UNet**.  
You can either implement your own or use an existing PyTorch implementation.  
A recommended repository is: [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)

---

## Part 3: Training the UNet

Once the dataset is loaded, you can:

- Define the **loss function** (e.g., `BCEWithLogitsLoss` or `DiceLoss`)
- Define the **optimizer** (e.g., Adam)
- Train the model for a number of epochs
- Save the trained model to a file (e.g., `unet_model.pth`)

Training involves iterating over the dataset from the DataLoader, computing predictions, calculating the loss, and updating the model weights.

---

## Part 4: Evaluation

After training, evaluate the model by:

- Running it on the validation or test dataset
- Applying a **sigmoid** or **softmax** to the outputs
- Converting predictions to binary masks
- Calculating metrics like **IoU** or **Dice coefficient**
- Optionally, visualize the predicted masks against the ground truth

---

## Going Further

- Apply **data augmentation** to increase dataset diversity
- Experiment with **different UNet variants or deeper models**
- Integrate multiple simulation runs to create a richer dataset
- Deploy the trained UNet for **real-time segmentation** in the Unity simulation