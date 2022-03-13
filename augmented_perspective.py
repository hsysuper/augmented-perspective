import argparse
import os
import numpy as np
import math
from skimage import io
import pathlib
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


def get_greyscale_img(img):
    """
    Get greyscale copy of this image
    :param img: (H, W, 3)
    :return: img: (H, W)
    """
    greyscale = np.array([0.299, 0.587, 0.114])
    img = img.astype(np.double)
    img = img.dot(greyscale)
    return img


def reprojection_inv_method(image, depth_map, M, RT):
    """
    Reproject using a translation matrix
    :param image: input RGB image (H, W)
    :param M: camera matrix
    :param T: translation matrix (3, 4)
    :return: reprojected image
    """
    print("Start reprojection")
    print("image.shape", image.shape)
    print("M.shape", M.shape)
    print("RT.shape", RT.shape)
    K_old = M[0:3, 0:3]
    K_inv = np.linalg.inv(K_old)
    RT_combined = np.zeros((3, 4), dtype=np.float64)
    RT_combined[0:3, 0:3] = RT[0:3, 0:3]
    RT_combined[:, 3] = M[:, 3]
    M_new = K_old.dot(RT_combined)
    H, W = image.shape
    new_image = np.zeros((H, W), dtype=np.float64)
    for h in range(H):
        for w in range(W):
            vector = np.array([h, w, 1], dtype=np.float64)
            vector -= M[:, 3]
            pseudo_3d = K_inv.dot(vector)
            pseudo_3d_vector = np.array([pseudo_3d[0], pseudo_3d[1], pseudo_3d[2], 1], dtype=np.float64)
            new_vector = M_new.dot(pseudo_3d_vector)
            new_vector /= new_vector[-1]
            new_vector[0] = min(max(H, new_vector[0]), 0)
            new_vector[1] = min(max(W, new_vector[1]), 0)
            new_image[int(new_vector[0]), int(new_vector[1])] = image[h, w]
    return new_image.astype(np.uint8)


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
    H, W = image.shape
    new_image = np.zeros((H, W), dtype=np.float64)

    px_to_depth_and_color = {}
    for u in range(W):
        for v in range(H):
            z = depth_map[v, u]
            img_h = np.array([u, v, 1, 1.0 / z], dtype=np.float64)
            world_p_h = z * M_inv.dot(img_h.T)
            world_p_h /= world_p_h[-1]
            p_new = M_new.dot(world_p_h)
            p_new /= p_new[-1]
            x_new = int(p_new[0])
            y_new = int(p_new[1])
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


if __name__ == '__main__':
    args = parse_args()
    image_path = args.image_path
    depth_map_path = args.depth_map_path

    output_name = pathlib.Path(depth_map_path).stem
    output_directory = pathlib.Path("outputs")

    image = io.imread(image_path)
    depth_map = np.load(depth_map_path)
    # M = calibrate(depth_map)
    # M = np.array([
    #     [7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01],
    #     [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01],
    #     [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03],
    # ])
    M = np.array([
        [7.070493e+02, 0.000000e+00, 6.040814e+02, 0],
        [0.000000e+00, 7.070493e+02, 1.805066e+02, 0],
        [0.000000e+00, 0.000000e+00, 1.000000e+00, 0],
    ])

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

    greyscale_img = get_greyscale_img(image)
    greyscale_img_path = os.path.join(output_directory, "{}_greyscale.jpeg".format(output_name))
    io.imsave(greyscale_img_path, greyscale_img.astype(np.uint8))

    new_image = reprojection(greyscale_img, depth_map, M, RT)

    reprojected_image_path = os.path.join(output_directory, "{}_reprojected.jpeg".format(output_name))
    print("Saving image {} to {}".format(new_image.shape, reprojected_image_path))
    io.imsave(reprojected_image_path, new_image)
