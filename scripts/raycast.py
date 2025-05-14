import math

from cv2 import Mat, imshow, imread, waitKey, bitwise_or


def check_hit_around(image: Mat, x:int, y: int, offset: int) -> bool:
    for i in range(int(x - offset // 2), int(x + offset // 2 + 1)):
        if i < 0 or i > image.shape[1] - 1:
            continue
        if image[y][i][0] == 255 and image[y][i][1] == 255 and image[y][i][2] == 255:
            return True
        if image[y][i][0] == 0 and image[y][i][1] == 0 and image[y][i][2] == 255:
            return True
    return False

def get_raycast(image: Mat, nb_ray: int, field_view: float) -> [float]:
    if nb_ray < 0:
        raise Exception('Number of ray invalid')

    angle_offset: float = field_view / (nb_ray - 1)
    step_size = 1

    distances = []
    for k in range(nb_ray):
        hit: bool = False
        x: float = image.shape[1] / 2
        y: float = image.shape[0] - 1
        hit_dist: int = 0
        while x >= 0 and x < image.shape[1] and y >= 0 and y < image.shape[0]:
            angle: float =  k * angle_offset * math.pi / 180 + angle_offset * math.pi / 180 * ((180 - field_view) / angle_offset / 2)
            rounded_x: int = math.floor(x)
            rounded_y: int = math.floor(y)
            if check_hit_around(image, rounded_x, rounded_y, 10):
                distances.append(hit_dist)
                hit = True
                break
            # if image[rounded_y][rounded_x][0] == 255 and image[rounded_y][rounded_x][1] == 255 and image[rounded_y][rounded_x][2] == 255:
            #     distances.append(hit_dist)
            #     hit = True
            #     break
            # if image[rounded_y][rounded_x][0] == 0 and image[rounded_y][rounded_x][1] == 0 and image[rounded_y][rounded_x][2] == 255:
            #     distances.append(hit_dist)
            #     hit = True
            #     break
            image[rounded_y][rounded_x] = [255, 0, 0]
            x += step_size * math.cos(angle)
            y -= step_size * math.sin(angle)
            hit_dist += 1
        if not hit:
            distances.append(hit_dist)
    # imshow("image", image)
    # waitKey(0)
    return distances

if __name__ == '__main__':
    import os
    img = imread(os.path.join('merged_result.png'))
    get_raycast(img, 10, 180)
    # for i in range(10, 51):
    #     img = imread(os.path.join('images', f'{i}_predicted_merged_mask.png'))
    #     print(get_raycast(img, 10, 100))
    # img1 = imread(os.path.join('Mask', '0-left.png'))
    # img2 = imread(os.path.join('Mask', '0-right.png'))
    # img = bitwise_or(img1, img2)

