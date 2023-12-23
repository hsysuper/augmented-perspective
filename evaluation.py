import argparse
import torch
import torch.nn.functional as F
import numpy as np
import math
import cv2

tensorify = lambda x: torch.Tensor(x.transpose(
    (2, 0, 1))).unsqueeze(0).float().div(255.0)


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)

    # Converting to 2D
    _2d_window = _1d_window.mm(
        _1d_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(
        _2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1,
         img2,
         val_range,
         window_size=11,
         window=None,
         size_average=True,
         full=False):

    L = val_range    # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height,
                        width)    # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad,
                         groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad,
                         groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad,
                       groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01)**2    # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03)**2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret


def SSIM(img1_path, img2_path, num_channels):
    if num_channels == 1:
        image_1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        image_2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    else:
        image_1 = cv2.imread(img1_path)    # For rgb images use default
        image_2 = cv2.imread(img2_path)    # For rgb images use default

    my_window = create_window(11, num_channels)

    if num_channels == 1:
        h, w = np.shape(image_1)
        image_1 = image_1.reshape(h, w, 1)
        image_2 = image_2.reshape(h, w, 1)
    _img1 = tensorify(image_1)
    _img2 = tensorify(image_2)

    return ssim(_img1, _img2, window=my_window, val_range=255)


def L2(img1, img2, num_channels):
    if num_channels == 1:
        image_1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
        image_2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    else:
        image_1 = cv2.imread(img1_path)    # For rgb images use default
        image_2 = cv2.imread(img2_path)    # For rgb images use default

    l2 = np.sum(np.power((image_1 - image_2), 2))
    return l2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluation Metric Selection.')
    parser.add_argument("eval_metric",
                        type=str,
                        help='name of the evaluation metric (L2, SSIM)')
    parser.add_argument("img1_path", type=str, help='path of the first image')
    parser.add_argument("img2_path", type=str, help='path of the second image')
    parser.add_argument(
        "num_channels",
        type=int,
        help='number of color channels (1 for greyscale 3 for RGB)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img1_path = args.img1_path
    img2_path = args.img2_path
    num_channels = args.num_channels
    if args.eval_metric == "L2":
        l2_score = L2(img1_path, img2_path, num_channels)
        print("L2 Score = ", l2_score)
    elif args.eval_metric == "SSIM":
        ssim_score = SSIM(img1_path, img2_path, num_channels)
        print("SSIM Score = ", ssim_score)
    else:
        print(
            "Usage: evaluation.py [L2 or SSIM] [img1_path] [img2_path] [num_channels]"
        )
