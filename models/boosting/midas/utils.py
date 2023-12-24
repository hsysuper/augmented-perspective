"""Utils for monoDepth.
"""
import numpy as np
import cv2


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def write_depth(path, depth, bits=1, colored=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    if colored == True:
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8 * bits)) - 1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    print("Saving depth image {} to {}".format(depth.shape, path + '.png'))
    if bits == 1 or colored:
        out = out.astype("uint8")
        if colored:
            out = cv2.applyColorMap(out, cv2.COLORMAP_INFERNO)
        cv2.imwrite(path + '.png', out)
    elif bits == 2:
        cv2.imwrite(path + '.png', out.astype("uint16"))

    # scale to 1 to 1000
    print("Saving raw depth data {} to {}".format(depth.shape,
                                                  path + '_raw.npy'))
    np.save(path + '_raw.npy', depth)
    scaled_disp, scaled_inverse_depth = disp_to_depth(depth, 1, 1000)
    print("Saving scaled_inverse_depth data {} to {}".format(
        scaled_inverse_depth.shape, path + '_inverse.npy'))
    np.save(path + '_inverse.npy', scaled_inverse_depth)
    scaled_depth = 1000 - 999 * (depth - depth_min) / (depth_max - depth_min)
    print("Saving depth data {} to {}".format(scaled_depth.shape,
                                              path + '.npy'))
    np.save(path + '.npy', scaled_depth)
    print(scaled_depth)
    return
