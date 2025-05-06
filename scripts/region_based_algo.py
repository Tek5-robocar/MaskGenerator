import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage import data, morphology, filters
from skimage.color import rgb2gray
from skimage.segmentation import watershed, mark_boundaries
from skimage.color import label2rgb
import scipy.ndimage as nd

# Set up display parameters
plt.rcParams["figure.figsize"] = (12, 8)
plt.style.use('ggplot')

# Load image and convert to grayscale
rocket = data.rocket()
rocket_wh = rgb2gray(rocket)

# Create figure for all plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
ax = axes.ravel()

# Display original image
ax[0].imshow(rocket)
ax[0].set_title('Original Image')
ax[0].axis('off')

# Display grayscale image
ax[1].imshow(rocket_wh, cmap='gray')
ax[1].set_title('Grayscale Image')
ax[1].axis('off')

# Canny Edge Detection
edges = canny(rocket_wh, sigma=1.5)
ax[2].imshow(edges, cmap='gray')
ax[2].set_title('Canny Edge Detection')
ax[2].axis('off')

# Region Filling
fill_im = nd.binary_fill_holes(edges)
ax[3].imshow(fill_im, cmap='gray')
ax[3].set_title('Region Filling')
ax[3].axis('off')

# Watershed Segmentation
elevation_map = filters.sobel(rocket_wh)

# Create markers
markers = np.zeros_like(rocket_wh)
markers[rocket_wh < rocket_wh.mean() - 0.1] = 1  # Dark regions
markers[rocket_wh > rocket_wh.mean() + 0.1] = 2  # Bright regions

# Apply watershed
segmentation = watershed(elevation_map, markers)
segmentation = nd.binary_fill_holes(segmentation - 1)

# Label regions
label_rock, num_labels = nd.label(segmentation)
print(f"Number of regions found: {num_labels}")

# Create overlay
image_label_overlay = label2rgb(label_rock, image=rocket_wh, bg_label=0)

# Display watershed results
ax[4].imshow(elevation_map, cmap='terrain')
ax[4].set_title('Elevation Map (Sobel)')
ax[4].axis('off')

ax[5].imshow(mark_boundaries(rocket_wh, label_rock))
ax[5].set_title('Watershed Segmentation')
ax[5].axis('off')

plt.tight_layout()
plt.show()

# Final comparison plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(rocket_wh, cmap='gray')
ax1.contour(segmentation, [0.5], linewidths=1.8, colors='red')
ax1.set_title('Original with Contours')
ax1.axis('off')

ax2.imshow(image_label_overlay)
ax2.set_title('Region Overlay')
ax2.axis('off')

plt.tight_layout()
plt.show()