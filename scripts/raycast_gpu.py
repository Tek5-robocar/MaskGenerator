import math
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt


class GPURaycaster:
    def __init__(self, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'

    def check_hit_around(self, image_tensor, x, y, offset=10):
        """
        GPU-accelerated version of check_hit_around
        """
        half_offset = offset // 2
        x_start = torch.clamp(x - half_offset, 0, image_tensor.shape[2] - 1)
        x_end = torch.clamp(x + half_offset + 1, 0, image_tensor.shape[2] - 1)

        # Check for white pixels (255, 255, 255) or red pixels (0, 0, 255)
        region = image_tensor[:, y:y + 1, x_start:x_end]
        white_hit = torch.all(region == torch.tensor([255, 255, 255], device=self.device), dim=0).any()
        red_hit = torch.all(region == torch.tensor([0, 0, 255], device=self.device), dim=0).any()

        return white_hit or red_hit

    def raycast(self, image, nb_ray=10, field_view=180):
        """
        GPU-accelerated raycasting
        Args:
            image: Input image (numpy array)
            nb_ray: Number of rays
            field_view: Field of view in degrees
        Returns:
            distances: List of hit distances
            debug_image: Image with rays drawn (numpy array)
        """
        if nb_ray <= 0:
            raise ValueError('Number of rays must be positive')

        # Convert image to PyTorch tensor and move to GPU
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(self.device)
        debug_image = image_tensor.clone()

        angle_offset = field_view / (nb_ray - 1) if nb_ray > 1 else 0
        step_size = 1
        center_x = image.shape[1] / 2
        start_y = image.shape[0] - 1

        distances = []

        for k in range(nb_ray):
            hit = False
            x = torch.tensor(center_x, device=self.device)
            y = torch.tensor(start_y, device=self.device)
            hit_dist = 0

            # Calculate angle for this ray
            angle_deg = k * angle_offset + ((180 - field_view) / 2)
            angle_rad = math.radians(angle_deg)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)

            while True:
                # Check bounds
                if x < 0 or x >= image.shape[1] or y < 0 or y >= image.shape[0]:
                    break

                rounded_x = x.long()
                rounded_y = y.long()

                # Check for hits
                if self.check_hit_around(image_tensor, rounded_x, rounded_y):
                    distances.append(hit_dist)
                    hit = True
                    break

                # Mark the ray path (red color)
                debug_image[:, rounded_y, rounded_x] = torch.tensor([255, 0, 0], device=self.device)

                # Move along the ray
                x += step_size * cos_angle
                y -= step_size * sin_angle
                hit_dist += 1

            if not hit:
                distances.append(hit_dist)

        # Convert debug image back to numpy
        debug_image = debug_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        return distances, debug_image


def main():
    # Initialize raycaster
    raycaster = GPURaycaster()

    # Load image
    img = cv2.imread('merged_result.png')
    if img is None:
        raise FileNotFoundError("Image not found")

    # Run raycasting
    distances, debug_img = raycaster.raycast(img, nb_ray=10, field_view=180)
    print("Distances:", distances)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    plt.title("Raycasting Result")
    plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()