import argparse
import importlib
import logging
import math
import pathlib
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


def get_argument_parser():
    """
    Arg parser for command line.
    """
    models_list = get_depth_model_list()
    parser = argparse.ArgumentParser(
        description="Augmented perspective module initialization.")
    parser.add_argument("--image_path",
                        type=pathlib.Path,
                        help="path to a test image",
                        default=None)
    parser.add_argument("--depth_map_path",
                        type=pathlib.Path,
                        help="path to depth map of test image",
                        default=None)
    parser.add_argument(
        "--depth_model",
        type=str,
        help=
        f"name of depth model used for creating depth map under models/, allowed = {models_list}",
        choices=models_list,
        default=None)
    parser.add_argument("--output_path",
                        type=pathlib.Path,
                        help="output path",
                        default=pathlib.Path("outputs"))
    parser.add_argument(
        "--save_filled_only",
        type=bool,
        help=
        "whether to only save the filled image without saving raw rotation only images",
        action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--angle",
        type=float,
        help=
        "angle in degree to rotate the image perspective clockwise around x-axis",
        default=15.0)
    parser.add_argument(
        "--translation",
        type=float,
        help=
        "linear translation of the camera, some translation is needed after rotation to keep image centered",
        default=-0.3)
    parser.add_argument(
        "--frames",
        type=int,
        help=
        "number of frames to rotate, 0 if only one frame is needed. As a result 0 and 1 has the same effect",
        default=0)
    parser.add_argument(
        "--scale_ratio",
        type=float,
        help="scaling ratio of the depth map during normalization",
        default=51)
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements up to DEBUG level",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.INFO)
    return parser


def normalize_depth_map(depth_map, scale_ratio: float):
    """
    Normalize the depth map

    :param depth_map: depth map input
    :param scale_ratio: scaling ratio in float
    :return:
    """
    normalized_depth_map = depth_map / scale_ratio
    return normalized_depth_map


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

    :param image: image array
    :return: filled image array
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


def run_augmented_perspective(argv):
    """
    Run augmented perspective algorithm to change perspective of the input image

    :param argv: input command line arguments array free sys.argv
    :return: None, save image to disk
    """
    sys.argv = argv
    parser = get_argument_parser()
    args = parser.parse_args()
    if not args.image_path:
        parser.error("--image_path is required")
        return
    if not args.depth_map_path:
        parser.error("--depth_map_path is required")
        return
    if not args.depth_model:
        parser.error("--depth_model is required")
        return

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        level=args.log_level,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True)

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
        logging.info("normalizing with scale_ratio={}".format(
            args.scale_ratio))
        depth_map = normalize_depth_map(depth_map, args.scale_ratio)

    logging.info(f"Getting intrinsic matrix for {args.image_path}")
    try:
        intrinsic_matrix_path = datasets.get_intrinsic_matrix(args.image_path)
        logging.info(
            f"Found intrinsic matrix file {intrinsic_matrix_path.relative_to(pathlib.Path.cwd())}"
        )
        M = np.loadtxt(str(intrinsic_matrix_path))
        M = M.reshape((3, 4))
    except Exception as e:
        logging.info(
            f"Could not find intrinsic matrix. Using calibrate(). Reason: {e}")
        logging.info(traceback.format_exc())
        M = calibrate(depth_map)

    if not args.frames:
        angles = [args.angle]
        translations = [args.translation]
    else:
        angles = np.linspace(0, args.angle, args.frames)
        translations = np.linspace(0, args.translation, args.frames)

    logging.info(f"Start re-projection for angles: {angles}")
    durations = []
    for i in range(len(angles)):
        logging.info(
            f"[{i+1}/{len(angles)}] Start reprojection at {angles[i]} degree clockwise around b axis"
        )
        start_time = time.time()

        # ROTATIONS
        a = math.pi * 0 / 180
        b = math.pi * angles[i] / 180
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

        # translations
        T = np.array([translations[i], 0, 0, 1], dtype=np.float64)

        RT = np.zeros((4, 4), dtype=np.float64)
        RT[0:3, 0:3] = R
        RT[:, 3] = T
        logging.info(f"RT:\n{RT}")

        # reprojection
        new_image = reprojection(image, depth_map, M, RT)

        # save raw image with black blocks
        suffix = "" if not args.frames else f"_{i}"
        if not args.save_filled_only:
            reprojected_image_path = args.output_path / pathlib.Path(
                f"{output_name}_reprojected{suffix}.png")
            logging.info(
                f"Saving image {new_image.shape} to {reprojected_image_path}")
            io.imsave(str(reprojected_image_path), new_image)
        else:
            logging.info("Skip saving unfilled raw reprojections")

        # save filled image with full pixels
        filled_new_image = fill(new_image)
        reprojected_filled_image_path = args.output_path / pathlib.Path(
            f"{output_name}_reprojected_filled{suffix}.png")
        logging.info(
            f"Saving filled image {filled_new_image.shape} to {reprojected_filled_image_path}"
        )
        io.imsave(str(reprojected_filled_image_path), filled_new_image)

        # statistics
        duration = time.time() - start_time
        logging.info(f"Time taken: {duration} seconds")
        durations.append(duration)
    logging.info(
        f"Average time per frame: {sum(durations) / len(durations)} seconds")


if __name__ == '__main__':
    run_augmented_perspective(sys.argv)
