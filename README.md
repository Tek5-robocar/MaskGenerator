# Overview

This page provides a high-level overview of the **MaskGenerator** project, which is split into two main components:

[Simulation](./Writerside/topics/Overview_Simulation.md)  
: The Unity-based environment used to generate images and masks for AI training.

[AI](./Writerside/topics/Overview_AI.md)  
: The AI component that uses the generated datasets to train a UNet model for image segmentation.

---

## What is the MaskGenerator?

The **MaskGenerator** project is designed to create realistic datasets from a Unity simulation and train AI models to understand and interpret images.

The simulation provides customizable tracks, decor, and camera systems, while the AI component leverages these datasets to learn to recognize different features, such as roads, lanes, and objects.

By combining simulation and AI training, the project enables rapid experimentation and evaluation of models in controlled 3D environments.

---

## Glossary

Here are some key terms you might encounter while working with this project:

<deflist>

  <def title="AI Model">
    A system that can learn from examples and make predictions.  
    In this project, the AI model learns to identify features in images using the datasets generated from the simulation.
  </def>

  <def title="Unity">
    A widely used real-time 3D development engine for games, simulations, and AI environments.  
    It provides tools for physics, rendering, animation, and scripting — making it ideal for building custom training environments.
  </def>

  <def title="Simulation">
    The Unity environment designed to generate realistic images and masks.  
    It includes tracks, decor, cameras, and post-processing systems to enhance dataset quality.
  </def>

  <def title="Dataset">
    A collection of images and corresponding masks generated from the simulation.  
    These are used to teach the AI model which features to identify.
  </def>

  <def title="Mask">
    A special type of image that highlights specific parts of a scene.  
    Masks are used by the AI model as a reference during training.
  </def>

  <def title="UNet">
    A neural network model used for image segmentation.  
    It learns to predict masks from input images, helping the AI understand visual scenes.
  </def>

  <def title="Post-Processing">
    Visual effects applied to images in the simulation to make datasets more realistic.  
    Includes effects like blur, grain, and color grading.
  </def>

</deflist>