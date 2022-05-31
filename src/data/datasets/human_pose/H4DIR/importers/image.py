import cv2
import torch
import numpy as np

# TODO: avoid the back and forth between torch and numpy
def get_depth_image_from_points(resolution, res_scale, points):
    depth_tensor = torch.zeros([1, 1, int(resolution), int(resolution)], dtype=torch.float32)
    width = resolution * res_scale
    height = resolution * res_scale
    depth_img = np.zeros([int(height), int(width), 1])
    points_np = points.cpu().numpy()
    points_np = np.asarray(sorted(points_np.squeeze(), key=lambda x: x[..., 2]))[::-1]

    for i in range(points_np.shape[0]):
        norm_x_value = points_np[i, 0]
        norm_y_value = points_np[i, 1]
        norm_depth_value = points_np[i, 2]
        y = int(norm_y_value * height) - 1
        x = int(norm_x_value * width) - 1
        offset = 0
        if (x > offset-1 and x < width-offset and y > offset-1 and y < height-offset):
            depth_img = cv2.circle(depth_img, (x, y), 2 * int(res_scale), float(norm_depth_value), -1)
        else:
            print("error")

    depth_img = cv2.resize(depth_img, (int(resolution), int(resolution)), interpolation=cv2.INTER_LINEAR_EXACT)
    depth_tensor[0, 0, ...] = torch.from_numpy(depth_img)
    return depth_tensor
