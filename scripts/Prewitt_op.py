import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read input image (replace with your image path)
input_image = cv2.imread('input_image.jpg')

# Display input image
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.show()

# Convert to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Convert to float32 for calculations
gray_image = np.float32(gray_image)

# Initialize filtered image
filtered_image = np.zeros_like(gray_image)

# Prewitt Operator Masks
Mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
My = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

# Edge detection process (ignoring 1-pixel border)
for i in range(gray_image.shape[0] - 2):
    for j in range(gray_image.shape[1] - 2):
        # Gradient approximations
        window = gray_image[i:i + 3, j:j + 3]
        Gx = np.sum(Mx * window)
        Gy = np.sum(My * window)

        # Magnitude of vector (placed at center pixel)
        filtered_image[i + 1, j + 1] = np.sqrt(Gx ** 2 + Gy ** 2)

# Convert to uint8 for display
filtered_image = np.uint8(filtered_image)

# Display filtered image
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image (Prewitt Edge Detection)')
plt.show()

# Thresholding (adjust thresholdValue as needed)
thresholdValue = 100
_, output_image = cv2.threshold(filtered_image, thresholdValue, 255, cv2.THRESH_BINARY)

# Display final edge-detected image
plt.imshow(output_image, cmap='gray')
plt.title('Edge Detected Image (Binary)')
plt.show()