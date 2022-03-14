import argparse
import os
import math
import pathlib
import sys
import time

import numpy as np
from skimage import io

from calibration import calibrate


def parse_args():
    """
    Arg parser for CMD line.
    """
    parser = argparse.ArgumentParser(description='Augmented perspective module initialization.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image',
                        required=True)
    parser.add_argument('--depth_map_path', type=str,
                        help='path to a test image depth map',
                        required=True)
    return parser.parse_args()


def reprojection(image, depth_map, M, RT):
    """
    Reproject using a translation matrix
    :param image: input greyscale image (H, W)
    :param depth_map depth map
    :param M: camera matrix
    :param RT: translation matrix (3, 4)
    :return: reprojected image
    """
    print("Start reprojection")
    M_square = np.zeros((4, 4), dtype=np.float64)
    M_square[:3, :] = M
    M_square[3, 3] = 1
    M_inv = np.linalg.inv(M_square)
    M_new = M.dot(RT)
    H, W, C = image.shape
    new_image = np.zeros((H, W, C), dtype=np.float64)

    px_to_depth_and_color = {}
    for u in range(W):
        for v in range(H):
            z = depth_map[v, u]
            img_h = np.array([u, v, 1, 1.0 / z], dtype=np.float64)
            world_p_h = z * M_inv.dot(img_h.T)
            world_p_h /= world_p_h[-1]
            p_new = M_new.dot(world_p_h)
            p_new /= p_new[-1]
            x_new = round(p_new[0])
            y_new = round(p_new[1])
            # Only allow in-frame pixels
            if x_new < 0 or x_new >= W or y_new < 0 or y_new >= H:
                continue

            # X-Y is reversed in row-first matrices
            # There may be multiple pixels in the 3D world that map to the same
            # new 2D pixel. We should save the one with the least depth as that
            # is what the camera would see. Everything else is occluded.
            if (y_new, x_new) not in px_to_depth_and_color or z < px_to_depth_and_color[(y_new, x_new)]["depth"]:
                px_to_depth_and_color[(y_new, x_new)] = {"depth": z, "color": image[v,u]}

    for k, v in px_to_depth_and_color.items():
        new_image[k[0], k[1]] = v["color"]

    return new_image.astype(np.uint8)


def fill(image):
    """
    Take all black pixels in image and fill them with average of surrounding
    non-black pixels.
    """
    H, W, C = image.shape

    def get_neighbors(u, v):
        nbrs = [
            (u-1, v-1), (u-1, v), (u-1, v+1), (u, v-1), (u, v+1), (u+1, v-1), (u+1, v), (u+1, v+1)
        ]
        for i in reversed(range(len(nbrs))):
            u, v = nbrs[i]
            if u < 0 or u >= H or v < 0 or v >= W:
                nbrs.pop(i)
            else:
                px = image[u][v]
                if not px.any():
                    nbrs.pop(i)
        return nbrs

    filled_new_image = np.copy(image)
    for u in range(H):
        for v in range(W):
            px = image[u][v]
            if not px.any():
                nbrs = get_neighbors(u, v)
                # Only smooth the black pixel if it has at least 4 non-black neighbors.
                if len(nbrs) >= 4:
                    colors = np.array([image[u][v] for u, v in nbrs])
                    filled_new_image[u][v] = np.average(colors)

    return filled_new_image.astype(np.uint8)


def run_augmented_perspective(argv):
    start_time = time.time()
    print("before", sys.argv)
    sys.argv = argv
    print("after", sys.argv)
    args = parse_args()
    image_path = args.image_path
    depth_map_path = args.depth_map_path

    output_name = pathlib.Path(depth_map_path).stem
    output_directory = pathlib.Path("outputs")
    final_output_directory = pathlib.Path("output_images")
    os.makedirs(final_output_directory, exist_ok=True)

    image = io.imread(image_path)
    depth_map = np.load(depth_map_path)
    print("image size", image.shape)
    print("depth map size", depth_map.shape)

    M_map = {
        "kitti1.png": "09_26",
        "kitti2.png": "09_28",
        "kitti3.png": "09_26",
        "kitti4.png": "09_26",
        "kitti5.png": "09_26",
    }

    try:
        M = np.loadtxt(f"intrinsic_matrices/kitti_calib_cam_to_cam_{M_map[pathlib.Path(image_path).name]}.txt")
        M = M.reshape((3, 4))
    except:
        print("NOTE: Could not find intrinsic matrix. Using calibrate() function.")
        M = calibrate(depth_map)

    # ROTATIONS
    a = math.pi * 0 / 180
    b = math.pi * 15 / 180
    g = math.pi * 0 / 180
    RX = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]], dtype=np.float64)
    RY = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]], dtype=np.float64)
    RZ = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]], dtype=np.float64)
    R = RZ.dot(RY.dot(RX))

    # TRANSLATIONS
    T = np.array([-0.3, 0, 0, 1], dtype=np.float64)

    RT = np.zeros((4, 4), dtype=np.float64)
    RT[0:3, 0:3] = R
    RT[:, 3] = T
    print("RT\n", RT)

    new_image = reprojection(image, depth_map, M, RT)
    filled_new_image = fill(new_image)

    reprojected_image_path = os.path.join(final_output_directory, "{}_reprojected.png".format(output_name))
    reprojected_filled_image_path = os.path.join(final_output_directory, "{}_filled.png".format(output_name))
    print("Saving image {} to {}".format(new_image.shape, reprojected_image_path))
    io.imsave(reprojected_image_path, new_image)
    io.imsave(reprojected_filled_image_path, filled_new_image)
    print("Time taken: {} seconds".format(time.time() - start_time))


if __name__ == '__main__':
    run_augmented_perspective(sys.argv)
