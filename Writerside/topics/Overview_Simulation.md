# Overview

This page introduces the **simulation** component of the MaskGenerator project.  
It explains how to use, customize, and extend the simulation, as well as how to generate datasets for AI training.

---

## What Is This?

This simulation was developed using **Unity** to generate training data and experiment with custom AI models.

Unity was chosen for its **ease of use**, **extensive documentation**, **built-in physics**, and **powerful post-processing capabilities**.

By leveraging Unity’s **post-processing library**, we can enhance the captured images by applying effects such as blur, grain, and color grading, making the datasets more **realistic** and suitable for training robust AI models.

By following the tutorial, you’ll learn how to:

- Run the simulation out of the box
- Customize tracks, decor, and camera systems
- Use the config file to generate high-quality image datasets

> 📘 If you want to try it yourself, follow the step-by-step setup in the [tutorial](Tutorial_Simulation.md).  
> 🔄 To understand how the AI connects and interacts with the simulation, check out the [AI Overview](Overview_AI.md).

---

## Glossary

<deflist>

  <def title="Unity">
    A widely used real-time 3D development engine for games, simulations, and AI environments.  
    It provides tools for physics, rendering, animation, and scripting — making it ideal for building custom training environments.
  </def>

  <def title="Simulation">
    The Unity environment designed to generate training data and test AI models.  
    It includes tracks, decor, cameras, and post-processing systems.
  </def>

  <def title="Dataset Generation">
    The process of automatically capturing images and masks from the simulation to create datasets used for training AI models.
  </def>

  <def title="Config File">
    A JSON file that defines parameters for dataset generation, such as image size, camera angles, motion blur, noise, and line width variations.
  </def>

  <def title="Tracks">
    Paths or routes within the simulation environment that agents, vehicles, or cameras follow.  
    Tracks are defined using Unity’s <b>LineRenderer</b> and are used to generate colliders and positioning data.
  </def>

  <def title="Decor">
    Static environmental elements — such as trees, signs, and objects — that enhance scene realism and can be toggled on or off.
  </def>

  <def title="Post-Processing">
    A set of visual effects applied to images during capture to simulate real-world camera imperfections.  
    Includes <b>motion blur</b>, <b>grain</b>, <b>color grading</b>, and <b>shapes</b> to make datasets more realistic.
  </def>

  <def title="Vision System">
    A Unity-based camera system responsible for capturing both RGB and mask images along the track.  
    It applies post-processing effects and follows paths based on the config file.
  </def>

  <def title="Mask">
    A binary or multi-class image used to label specific features (e.g., road lines, background, decor).  
    Masks are generated alongside RGB images for training AI models.
  </def>

  <def title="Game Manager">
    The main controller of the simulation, responsible for loading configs, spawning agents, controlling the vision system, and managing dataset generation.
  </def>

</deflist>