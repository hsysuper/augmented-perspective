import os

import cv2
import numpy as np
import skimage.measure
from six.moves import urllib

# miscellaneous function for reading, writing and processing rgb and depth images.


def resize_with_pool(img, size):
    i_size = img.shape[0]
    n = int(np.floor(i_size / size))

    out = skimage.measure.block_reduce(img, (n, n), np.max)
    return out


def read_image(path):
    img = cv2.imread(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    return img


def generate_mask(size):
    """
    Generates a Gaussian mask

    :param size:
    :return:
    """
    mask = np.zeros(size, dtype=np.float32)
    sigma = int(size[0] / 16)
    k_size = int(2 * np.ceil(2 * int(size[0] / 16)) + 1)
    mask[int(0.15 * size[0]):size[0] - int(0.15 * size[0]),
         int(0.15 * size[1]):size[1] - int(0.15 * size[1])] = 1
    mask = cv2.GaussianBlur(mask, (int(k_size), int(k_size)), sigma)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = mask.astype(np.float32)
    return mask


def extract_image_patch(image, rect):
    """
    Extract the given patch pixels from a given image.

    :param image:
    :param rect:
    :return:
    """
    w1 = rect[0]
    h1 = rect[1]
    w2 = w1 + rect[2]
    h2 = h1 + rect[3]
    image_patch = image[h1:h2, w1:w2]
    return image_patch


def get_gradient_density_from_integral(integral_image, rect):
    """
    Computes the gradient density of a given patch from the gradient integral image.

    :param integral_image:
    :param rect:
    :return:
    """
    x1 = rect[1]
    x2 = rect[1] + rect[3]
    y1 = rect[0]
    y2 = rect[0] + rect[2]
    value = integral_image[x2, y2] - integral_image[x1, y2] - integral_image[
        x2, y1] + integral_image[x1, y1]
    return value


def rgb2gray(rgb):
    """
    Converts rgb to gray

    :param rgb: input RGB image
    :return: grayscale image
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def calculate_processing_res(img,
                             base_size,
                             confidence=0.1,
                             scale_threshold=3,
                             whole_size_threshold=3000):
    """
    Returns the R_x resolution described in section 5 of the main paper.

    :param img: input rgb image
    :param base_size: size the dilation kernel which is equal to receptive field of the network.
    :param confidence: value of x in R_x; allowed percentage of pixels that are not getting any contextual cue.
    :param scale_threshold: maximum allowed upscaling on the input image ; it has been set to 3.
    :param whole_size_threshold: maximum allowed resolution. (R_max from section 6 of the main paper)
    :return:
        output_size_scale * speed_scale: The computed R_x resolution
        patch_scale: K parameter from section 6 of the paper
    """
    # speed scale parameter is to process every image in a smaller size to accelerate the R_x resolution search
    speed_scale = 32
    image_dim = int(min(img.shape[0:2]))

    gray = rgb2gray(img)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)) + np.abs(
        cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad = cv2.resize(grad, (image_dim, image_dim), cv2.INTER_AREA)

    # thresholding the gradient map to generate the edge-map as a proxy of the contextual cues
    m = grad.min()
    M = grad.max()
    middle = m + (0.4 * (M - m))
    grad[grad < middle] = 0
    grad[grad >= middle] = 1

    # dilation kernel with size of the receptive field
    kernel = np.ones(
        (int(base_size / speed_scale), int(base_size / speed_scale)), float)
    # dilation kernel with size of the a quarter of receptive field used to compute k
    # as described in section 6 of main paper
    kernel2 = np.ones(
        (int(base_size / (4 * speed_scale)), int(base_size /
                                                 (4 * speed_scale))), float)

    # Output resolution limit set by the whole_size_threshold and scale_threshold.
    threshold = min(whole_size_threshold, scale_threshold * max(img.shape[:2]))

    output_size_scale = base_size / speed_scale
    for p_size in range(int(base_size / speed_scale),
                        int(threshold / speed_scale),
                        int(base_size / (2 * speed_scale))):
        grad_resized = resize_with_pool(grad, p_size)
        grad_resized = cv2.resize(grad_resized, (p_size, p_size),
                                  cv2.INTER_NEAREST)
        grad_resized[grad_resized >= 0.5] = 1
        grad_resized[grad_resized < 0.5] = 0

        dilated = cv2.dilate(grad_resized, kernel, iterations=1)
        meanvalue = (1 - dilated).mean()
        if meanvalue > confidence:
            break
        else:
            output_size_scale = p_size

    grad_region = cv2.dilate(grad_resized, kernel2, iterations=1)
    patch_scale = grad_region.mean()

    return int(output_size_scale * speed_scale), patch_scale


def apply_grid_patch(blsize, stride, img, box):
    """
    Extract a simple grid patch.

    :param blsize:
    :param stride:
    :param img:
    :param box:
    :return:
    """
    counter1 = 0
    patch_bound_list = {}
    for k in range(blsize, img.shape[1] - blsize, stride):
        for j in range(blsize, img.shape[0] - blsize, stride):
            patch_bound_list[str(counter1)] = {}
            patchbounds = [
                j - blsize, k - blsize, j - blsize + 2 * blsize,
                k - blsize + 2 * blsize
            ]
            patch_bound = [
                box[0] + patchbounds[1], box[1] + patchbounds[0],
                patchbounds[3] - patchbounds[1],
                patchbounds[2] - patchbounds[0]
            ]
            patch_bound_list[str(counter1)]['rect'] = patch_bound
            patch_bound_list[str(counter1)]['size'] = patch_bound[2]
            counter1 = counter1 + 1
    return patch_bound_list


class Images:

    def __init__(self, root_dir, files, index):
        self.root_dir = root_dir
        name = files[index]
        self.rgb_image = read_image(os.path.join(self.root_dir, name))
        name = name.replace(".jpg", "")
        name = name.replace(".png", "")
        name = name.replace(".jpeg", "")
        self.name = name


class ImageAndPatches:

    def __init__(self, root_dir, name, patches_info, rgb_image, scale=1):
        self.root_dir = root_dir
        self.patches_info = patches_info
        self.name = name
        self.patches = patches_info
        self.scale = scale

        self.rgb_image = cv2.resize(rgb_image, (round(
            rgb_image.shape[1] * scale), round(rgb_image.shape[0] * scale)),
                                    interpolation=cv2.INTER_CUBIC)

        self.do_have_estimate = False
        self.estimation_updated_image = None
        self.estimation_base_image = None

    def __len__(self):
        return len(self.patches)

    def set_base_estimate(self, est):
        self.estimation_base_image = est
        if self.estimation_updated_image is not None:
            self.do_have_estimate = True

    def set_updated_estimate(self, est):
        self.estimation_updated_image = est
        if self.estimation_base_image is not None:
            self.do_have_estimate = True

    def __getitem__(self, index):
        patch_id = int(self.patches[index][0])
        rect = np.array(self.patches[index][1]['rect'])
        msize = self.patches[index][1]['size']

        # applying scale to rect:
        rect = np.round(rect * self.scale)
        rect = rect.astype('int')
        msize = round(msize * self.scale)

        patch_rgb = extract_image_patch(self.rgb_image, rect)
        if self.do_have_estimate:
            patch_whole_estimate_base = extract_image_patch(
                self.estimation_base_image, rect)
            patch_whole_estimate_updated = extract_image_patch(
                self.estimation_updated_image, rect)
            return {
                'patch_rgb': patch_rgb,
                'patch_whole_estimate_base': patch_whole_estimate_base,
                'patch_whole_estimate_updated': patch_whole_estimate_updated,
                'rect': rect,
                'size': msize,
                'id': patch_id
            }
        else:
            return {
                'patch_rgb': patch_rgb,
                'rect': rect,
                'size': msize,
                'id': patch_id
            }


class ImageDataset:

    def __init__(self, root_dir, files=None):
        self.dataset_dir = root_dir
        self.rgb_image_dir = root_dir
        if not files:
            self.files = sorted(os.listdir(self.rgb_image_dir))
        else:
            self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return Images(self.rgb_image_dir, self.files, index)


def download_model_if_doesnt_exist(model_name):
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "merge":
        "https://sfu.ca/~yagiz/CVPR21/latest_net_G.pth",
        "midas":
        "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
    }

    model_url = download_paths[model_name]

    if model_name == "merge":
        model_path = os.path.join("pix2pix", "checkpoints", "mergemodel")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # see if we have the model already downloaded...
        model_path = os.path.join(model_path, "latest_net_G.pth")
        if not os.path.exists(model_path):
            print("-> Downloading pretrained MergeModel to {}".format(
                model_path))
            urllib.request.urlretrieve(model_url, model_path)

    elif model_name == "midas":
        model_path = "midas"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # see if we have the model already downloaded...
        model_path = os.path.join(model_path, "model.pt")
        if not os.path.exists(model_path):
            print("-> Downloading pretrained MidasModel to {}".format(
                model_path))
            urllib.request.urlretrieve(model_url, model_path)
