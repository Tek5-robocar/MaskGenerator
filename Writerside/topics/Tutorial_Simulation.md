# Tutorial

> **Note**: This tutorial assumes you have some basic knowledge of Unity.  
> It **does not** explain how to use Unity from scratch, and may be challenging for complete beginners.

### In this tutorial, you will learn how to:

- Use the simulation as-is
- Customize the simulation
- Generate datasets using a config file

---

## Before You Start

Ensure the following prerequisites are met:

- Unity Hub is installed (download it [here](https://unity.com/download))
- The **MaskGenerator** Unity project folder is accessible and fully unzipped
- You have a stable internet connection

---

## Part 1: Open the Project

Follow these steps to open the project and install required packages:

1. Open Unity Hub and **Add the project from disk**  
   ![Add project in Unity Hub](add_in_hub.png)
2. Install the required Unity Editor version if prompted
3. Open the project
4. Load the scene:  
   ![Open scene](open_scene_racingsimulator.png)
    - Go to the **Assets** folder in the **Project** tab
    - Open the **Scenes** folder
    - Double-click on `RacingSimulator.unity`
5. Install required packages via the **Package Manager**  
   ![Open Unity Package Manager](package_manager_unity.png)
6. Ensure the following packages are installed:
    - `com.unity.ugui` (Unity UI)
    - `com.unity.ml-agents` (ML Agents)

If everything is set up correctly, you can now run the simulation using the **Play** button.

> **Tip**: If you cloned the project and textures are missing, you may need to use [Git LFS](https://git-lfs.com/)

---

## Part 2: Project Overview

This section gives a high-level description of key GameObjects and systems.

<deflist collapsible="true">

<def title="Decor" default-state="collapsed">
The <b>Decor</b> consists of various static environmental elements.  
![Decor](decor_unity.png)
</def>

<def title="Tracks" default-state="collapsed">
The <b>Tracks</b> GameObject contains all available tracks.  
To add a new track, place it under this object.

Each track includes:

- An inactive original line from the package
- A `LineRenderer` generated from that line

The `LineRenderer` allows direct access to positions and automatically generates colliders using the <b>Line Renderer Collider Generator</b> script.
</def>

<def title="UI" default-state="collapsed">
Located under the <b>Canvas</b> GameObject, the UI includes the following elements in the "OpenPanel":

1. <b>ViewDropDown</b> – switches camera views  
2. <b>TrackDropDown</b> – selects the active track  
3. <b>TrackBestScore</b> – displays best lap time (`-1:-1` if none exists)  
4. <b>Decor</b> – toggles decor visibility via the <b>Decor Button</b> script  
5. <b>RaycastVisionage</b> – toggles raycast visibility
</def>

<def title="Game Manager" default-state="collapsed">
Handles the main logic via:

- <b>Config Loader</b> – loads the config file, spawns agents, and wires components  
- <b>Raycast</b> – performs 2D raycasting and integrates with the UI
</def>

<def title="Central Line" default-state="collapsed">
A <b>Central Line</b> is generated when selecting a track.  
Its colliders are used to:

- Track agent progress
- Provide reward metrics (e.g., distance from center, randomized starts)
</def>

<def title="Post Processing" default-state="collapsed">
The <b>PostFX</b> GameObject manages post-processing effects, configured at runtime by the <b>Vision</b> script.

Currently supported effects include:

- <b>ColorGrading</b> – adjusts image tone and color  
- <b>Grain</b> – adds noise to simulate real-world imperfections  
- <b>MotionBlur</b> – mimics blur due to motion in real-world images
</def>

<def title="Vision" default-state="collapsed">
This is the core image generation system. It moves along the central line using an offset defined in the config file.  
It:

- Takes pictures with its camera
- Applies post-processing via <b>PostFX</b>
- Saves both RGB and mask images using a masking plane (its child object)
</def>

<def title="Camera" default-state="collapsed">
A top-down camera providing a map-style view used as a background for the UI.
</def>

</deflist>

---

## Part 3: Code Documentation

For more details about the scripts and components, please refer to the internal documentation provided in the codebase.

---

## Part 4: Generate a Dataset

To generate a dataset:

1. Open `generator-config.json` and configure the desired parameters.
2. Click **Play** in the Unity Editor to start the simulation.
3. The console will display progress (e.g., how many images have been generated).
4. When finished, the version’s output folder will open automatically with the generated dataset.

Here is an overview of the config file:

```json
{
   "nbImage": 10, // number of image pairs (RGB and mask)
   "rotationRange": 70, // random rotation range in degrees per frame (e.g., -70° to 70°), to avoid identical directions
   "imageWidth": 256, // width of each image
   "imageHeight": 256, // height of each image
   "posZoneRadius": 0, // radius within which the camera position is randomly offset to avoid always being centered
   "cameraHeightRange": { // vertical range for camera position to generalize perspectives
      "min": 2,
      "max": 3
   },
   "cameraAngleRange": { // downward tilt range of the camera, affects line appearance and position
      "min": 5,
      "max": 20
   },
   "blurQuantityPercent": 0, // amount of motion blur applied (in percent)
   "grainQuantityPercent": 0, // amount of grain (noise) added to the image
   "coloredGrainQuantityPercent": 0, // percentage of grain that is colored (e.g., 50% grain, 50% colored = 25% of images with colored grain)
   "colorGradientQuantityPercent": 0, // intensity of color grading (whitening/lighting effects)
   "shapeQuantityPercent": 0, // percentage of random white shapes to simulate light patterns on the ground
   "maxShapeDensity": 0, // how dense the shapes should be (higher = more shapes)
   "lineWidthMultiplierRange": { // range to randomly scale line widths per frame
      "min": 0.1,
      "max": 0.3
   },
   "versionDirectory": "0.1.2", // output directory name for this dataset version
   "fixMaskLineWidth": false // whether to fix mask line width (true = constant size; helps model learn consistent features)
}
```

---

## Contributing

If you encounter a bug or have trouble using the project, please [create an issue](#) to report it.

We welcome contributions! You can:

- Submit a pull request to fix bugs
- Improve or expand the documentation
- Add new features or enhancements

Thank you for helping make this project better!

---