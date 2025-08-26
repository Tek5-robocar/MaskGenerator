# Overview

This page introduces the **UNet Training** component of the MaskGenerator project.  
It explains how to load datasets generated from the simulation, train a UNet model, and evaluate its performance for image segmentation tasks.

---

## What Is This?

This pipeline leverages datasets generated from **Unity simulations** to train a **UNet** for semantic segmentation.

Using existing RGB images and corresponding masks, the UNet learns to predict segmentation masks from input images, enabling applications such as:

- Road and lane detection
- Object masking
- Scene understanding for AI agents

By following the tutorial, you’ll learn how to:

- Load datasets using PyTorch `DataLoader`
- Preprocess images and masks for training
- Train a UNet model for segmentation
- Evaluate predictions against ground truth masks

> 📘 For step-by-step instructions, follow the [tutorial](Tutorial_UNet.md).  
> 🔄 To see how this integrates with the simulation, check out the [Simulation Overview](Overview_Simulation.md).

---

## Glossary

<deflist>

  <def title="UNet">
    A convolutional neural network architecture commonly used for image segmentation.  
    It consists of an encoder-decoder structure with skip connections, enabling precise localization and context understanding.  
    Reference implementation: [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
  </def>

  <def title="Dataset">
    A collection of RGB images and corresponding masks generated from the Unity simulation.  
    Used as input-output pairs for training the UNet model.
  </def>

  <def title="DataLoader">
    A PyTorch utility to efficiently load and batch datasets during training and evaluation.  
    Supports shuffling, parallel loading, and transformations such as normalization.
  </def>

  <def title="Segmentation Mask">
    An image labeling different regions or objects of interest.  
    Masks can be binary or multi-class and are used as ground truth for model training.
  </def>

  <def title="Training">
    The process of optimizing the UNet weights using the dataset to minimize a loss function, such as BCE or Dice loss, so the model can accurately predict segmentation masks.
  </def>

  <def title="Evaluation">
    Assessing the model’s performance using metrics like Intersection over Union (IoU) or Dice coefficient, and optionally visualizing predicted masks against ground truth.
  </def>

  <def title="Preprocessing">
    Steps applied to images and masks before feeding them into the model.  
    May include resizing, normalization, and data augmentation to improve model robustness.
  </def>

</deflist>