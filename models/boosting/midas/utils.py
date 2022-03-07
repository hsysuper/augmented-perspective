"""Utils for monoDepth.
"""
import numpy as np
import cv2


def write_depth(path, depth, bits=1 , colored=False):
    """Write depth map to pfm and png file.

    Args:
        path (str): filepath without extension
        depth (array): depth
    """
    if colored == True:
        bits = 1

    depth_min = depth.min()
    depth_max = depth.max()

    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0

    if bits == 1 or colored:
        out = out.astype("uint8")
        if colored:
            out = cv2.applyColorMap(out,cv2.COLORMAP_INFERNO)
        cv2.imwrite(path+'.png', out)
    elif bits == 2:
        cv2.imwrite(path+'.png', out.astype("uint16"))

    return
