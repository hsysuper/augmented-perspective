import argparse
import logging
import math
import pathlib
import importlib
import sys
import time
import traceback

import numpy as np
from skimage import io

import datasets

from calibration import calibrate
from depth_model import get_depth_model_list
"""
Usage:
    python -m augmented_perspective --image_path <path-to-image> --depth_model <model-name>
"""


def parse_args():
    """
    Arg parser for command line.
    """
    models_list = get_depth_model_list()
    parser = argparse.ArgumentParser(
        description="Augmented perspective module initialization.")
    parser.add_argument("--image_path",
                        type=pathlib.Path,
                        help="path to a test image",
                        required=True)
    parser.add_argument("--depth_map_path",
                        type=pathlib.Path,
                        help="path to depth map of test image",
                        required=True)
    parser.add_argument(
        "--depth_model",
        type=str,
        help=
        f"name of depth model used for creating depth map under models/, allowed = {models_list}",
        choices=models_list,
        required=True)
    parser.add_argument("--output_path",
                        type=pathlib.Path,
                        help="output path",
                        default=pathlib.Path("outputs"))
    parser.add_argument("--intrinsic_matrix",
                        type=pathlib.Path,
                        help="Camera intrinsic matrix")
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements up to DEBUG level",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.INFO)
    return parser.parse_args()


def normalize_depth_map(depth, scale_ratio):
    normalized_depth = depth / scale_ratio
    return normalized_depth


def reprojection(image, depth_map, M, RT):
    """
    Reproject using a translation matrix
    :param image: input greyscale image (H, W)
    :param depth_map depth map
    :param M: camera matrix
    :param RT: translation matrix (3, 4)
    :return: reprojected image
    """
    logging.info("Start reprojection")
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
            if (y_new, x_new
                ) not in px_to_depth_and_color or z < px_to_depth_and_color[(
                    y_new, x_new)]["depth"]:
                px_to_depth_and_color[(y_new, x_new)] = {
                    "depth": z,
                    "color": image[v, u]
                }

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
        nbrs = [(u - 1, v - 1), (u - 1, v), (u - 1, v + 1), (u, v - 1),
                (u, v + 1), (u + 1, v - 1), (u + 1, v), (u + 1, v + 1)]
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


def run_augmented_perspective(
    argv,
    save_filled_only=False,
    ANGLE=15,
    TRANSLATION=-0.3,
    FRAMES=0,
    SCALE_RATIO=51,
):
    sys.argv = argv
    args = parse_args()

    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
        level=args.log_level,
        datefmt='%Y-%m-%d %H:%M:%S')

    image_name = args.image_path.stem
    output_name = f"{image_name}_{args.depth_model}"
    if not args.depth_map_path:
        args.depth_map_path = f"outputs/{output_name}_depth.npy"

    args.output_path.mkdir(parents=True, exist_ok=True)

    image = io.imread(args.image_path)
    depth_map = np.load(args.depth_map_path)

    logging.info(f"Optional steps for depth mode {args.depth_model}")
    depth_model_module = importlib.import_module(
        f"models.{args.depth_model}.depth_prediction")
    depth_model = depth_model_module.DepthModel
    if depth_model.require_normalization():
        logging.info("normalizing with scale_ratio={}".format(SCALE_RATIO))
        depth_map = normalize_depth_map(depth_map, SCALE_RATIO)

    logging.info(f"Getting intrinsic matrix for {args.image_path}")
    try:
        intrinsic_matrix_path = datasets.get_intrinsic_matrix(args.image_path)
        logging.info(f"Found intrinsic matrix file {intrinsic_matrix_path}")
        M = np.loadtxt(str(intrinsic_matrix_path))
        M = M.reshape((3, 4))
    except Exception as e:
        logging.info(
            f"NOTE: Could not find intrinsic matrix. Using calibrate() function. Reason: {e}"
        )
        logging.info(traceback.format_exc())
        M = calibrate(depth_map)

    if not FRAMES:
        ANGLES = [ANGLE]
        TRANSLATIONS = [TRANSLATION]
    else:
        ANGLES = np.linspace(0, ANGLE, FRAMES)
        TRANSLATIONS = np.linspace(0, TRANSLATION, FRAMES)

    logging.info(f"Start re-projection for angles: {ANGLES}")
    durations = []
    for i in range(len(ANGLES)):
        logging.info(
            f"Start re-projection at {ANGLES[i]} degree clockwise around b axis"
        )
        start_time = time.time()

        # ROTATIONS
        a = math.pi * 0 / 180
        b = math.pi * ANGLES[i] / 180
        g = math.pi * 0 / 180
        RX = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)],
                       [0, math.sin(a), math.cos(a)]],
                      dtype=np.float64)
        RY = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0],
                       [-math.sin(b), 0, math.cos(b)]],
                      dtype=np.float64)
        RZ = np.array([[math.cos(g), -math.sin(g), 0],
                       [math.sin(g), math.cos(g), 0], [0, 0, 1]],
                      dtype=np.float64)
        R = RZ.dot(RY.dot(RX))

        # TRANSLATIONS
        T = np.array([TRANSLATIONS[i], 0, 0, 1], dtype=np.float64)

        RT = np.zeros((4, 4), dtype=np.float64)
        RT[0:3, 0:3] = R
        RT[:, 3] = T
        logging.info(f"RT: {RT}\n")

        new_image = reprojection(image, depth_map, M, RT)
        filled_new_image = fill(new_image)

        suffix = "" if not FRAMES else f"_{i}"
        reprojected_image_path = args.output_path, pathlib.Path(
            f"{output_name}_reprojected{suffix}.png")
        reprojected_filled_image_path = args.output_path / pathlib.Path(
            f"{output_name}_reprojected_filled{suffix}.png")
        if not save_filled_only:
            logging.info(
                f"Saving image {new_image.shape} to {reprojected_image_path}")
            io.imsave(str(reprojected_image_path), new_image)
        logging.info(
            f"Saving filled image {filled_new_image.shape} to {reprojected_filled_image_path}"
        )
        io.imsave(str(reprojected_filled_image_path), filled_new_image)
        duration = time.time() - start_time
        logging.info(f"Time taken: {duration} seconds")
        durations.append(duration)
    logging.info(
        f"Average time per frame: {sum(durations) / len(durations)} seconds")


if __name__ == '__main__':
    run_augmented_perspective(sys.argv)
