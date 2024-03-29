import os
import time
import warnings
from operator import getitem

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose

from models.base_model import BaseDepthModel
# MIDAS
from .midas import utils as midas_utils
from .midas.models.midas_net import MidasNet
from .midas.models.transforms import Resize, NormalizeImage, PrepareForNet
from .pix2pix.models.pix2pix4depth_model import Pix2Pix4DepthModel
# PIX2PIX : MERGE NET
from .pix2pix.options.test_options import TestOptions
# OUR
from .utils import ImageAndPatches, ImageDataset, generate_mask, get_gradient_density_from_integral, calculate_processing_res, rgb2gray, \
    apply_grid_patch, download_model_if_doesnt_exist

warnings.simplefilter('ignore', np.RankWarning)

device = None

# Global variables
pix2pix_model: Pix2Pix4DepthModel = None
midas_model: MidasNet = None
factor = 0.0
whole_size_threshold = 3000    # R_max from the paper
GPU_threshold = 1600 - 32    # Limit for the GPU (NVIDIA RTX 2080), can be adjusted


# MAIN PART OF OUR METHOD
def boosting_get_depth_map(option, parser):
    print("Reading images from {}".format(option.image_path))
    dataset = ImageDataset(option.image_path, option.image_files)

    # select device
    global device
    device = torch.device(option.device)
    print("device: %s" % device)
    # Load merge network
    options_parser = TestOptions()
    options_parser.initialize(parser)
    opt = options_parser.parse()
    global pix2pix_model
    pix2pix_model = Pix2Pix4DepthModel(opt)
    pix2pix_model.save_dir = './pix2pix/checkpoints/mergemodel'
    download_model_if_doesnt_exist('merge')
    pix2pix_model.load_networks('latest')
    pix2pix_model.eval()

    # Use Midas depth estimation model
    download_model_if_doesnt_exist('midas')
    midas_model_path = "midas/model.pt"
    global midas_model
    midas_model = MidasNet(midas_model_path, non_negative=True)
    midas_model.to(device)
    midas_model.eval()

    # Generating required directories
    result_dir = option.output_path
    os.makedirs(result_dir, exist_ok=True)

    # Generate mask used to smoothly blend the local pathc estimations to the base estimate.
    # It is arbitrarily large to avoid artifacts during rescaling for each crop.
    mask_org = generate_mask((3000, 3000))
    mask = mask_org.copy()

    # Value x of R_x defined in the section 5 of the main paper.
    r_threshold_value = 0.2

    # Go through all images in input directory
    print("start processing")
    for image_ind, images in enumerate(dataset):
        print('processing image', image_ind, ':', images.name)
        start_time = time.time()

        # Load image from dataset
        img = images.rgb_image
        input_resolution = img.shape

        scale_threshold = 3    # Allows up-scaling with a scale up to 3

        # Find the best input resolution R-x. The resolution search described in section 5-double estimation of the
        # main paper and section B of the supplementary material.
        whole_image_optimal_size, patch_scale = calculate_processing_res(
            img, option.net_receptive_field_size, r_threshold_value,
            scale_threshold, whole_size_threshold)

        print('\t wholeImage being processed in :', whole_image_optimal_size)

        # Generate the base estimate using the double estimation.
        whole_estimate = double_estimate(img, option.net_receptive_field_size,
                                         whole_image_optimal_size,
                                         option.pix2pixsize)

        # Compute the multiplier described in section 6 of the main paper to make sure our initial patch can select
        # small high-density regions of the image.
        global factor
        factor = max(
            min(
                1, 4 * patch_scale * whole_image_optimal_size /
                whole_size_threshold), 0.2)
        print('Adjust factor is:', 1 / factor)

        # Check if Local boosting is beneficial.
        if option.max_res < whole_image_optimal_size:
            print("No Local boosting. Specified Max Res is smaller than R20")
            path = os.path.join(result_dir, f"{images.name}_boosting_disp")
            midas_utils.write_depth(
                path,
                cv2.resize(whole_estimate,
                           (input_resolution[1], input_resolution[0]),
                           interpolation=cv2.INTER_CUBIC),
                bits=2,
                colored=True)
            continue

        # Compute the default target resolution.
        if img.shape[0] > img.shape[1]:
            a = 2 * whole_image_optimal_size
            b = round(2 * whole_image_optimal_size * img.shape[1] /
                      img.shape[0])
        else:
            a = round(2 * whole_image_optimal_size * img.shape[0] /
                      img.shape[1])
            b = 2 * whole_image_optimal_size
        b = int(round(b / factor))
        a = int(round(a / factor))

        # recompute a, b and saturate to max res.
        if max(a, b) > option.max_res:
            print(
                'Default Res is higher than max-res: Reducing final resolution'
            )
            if img.shape[0] > img.shape[1]:
                a = option.max_res
                b = round(option.max_res * img.shape[1] / img.shape[0])
            else:
                a = round(option.max_res * img.shape[0] / img.shape[1])
                b = option.max_res
            b = int(b)
            a = int(a)

        img = cv2.resize(img, (b, a), interpolation=cv2.INTER_CUBIC)

        # Extract selected patches for local refinement
        base_size = option.net_receptive_field_size * 2
        patch_set = generate_patches(img, base_size)

        print('Target resolution: ', img.shape)

        mergein_scale = input_resolution[0] / img.shape[0]
        print('Dynamically change merged-in resolution; scale:', mergein_scale)

        image_and_patches = ImageAndPatches(option.image_path, images.name,
                                            patch_set, img, mergein_scale)
        whole_estimate_resized = cv2.resize(
            whole_estimate, (round(img.shape[1] * mergein_scale),
                             round(img.shape[0] * mergein_scale)),
            interpolation=cv2.INTER_CUBIC)
        image_and_patches.set_base_estimate(whole_estimate_resized.copy())
        image_and_patches.set_updated_estimate(whole_estimate_resized.copy())

        print('\t Resulted depth map res will be :',
              whole_estimate_resized.shape[:2])
        print('patches to process: ' + str(len(image_and_patches)))

        # Enumerate through all patches, generate their estimations and refining the base estimate.
        for patch_ind in range(len(image_and_patches)):

            # Get patch information
            patch = image_and_patches[patch_ind]    # patch object
            patch_rgb = patch['patch_rgb']    # rgb patch
            patch_whole_estimate_base = patch[
                'patch_whole_estimate_base']    # corresponding patch from base
            rect = patch['rect']    # patch size and location
            patch_id = patch['id']    # patch ID
            org_size = patch_whole_estimate_base.shape    # the original size from the unscaled input
            print('\t processing patch', patch_ind, '|', rect)

            # We apply double estimation for patches. The high resolution value is fixed to twice the receptive
            # field size of the network for patches to accelerate the process.
            patch_estimation = double_estimate(patch_rgb,
                                               option.net_receptive_field_size,
                                               option.patch_netsize,
                                               option.pix2pixsize)

            patch_estimation = cv2.resize(
                patch_estimation, (option.pix2pixsize, option.pix2pixsize),
                interpolation=cv2.INTER_CUBIC)

            patch_whole_estimate_base = cv2.resize(
                patch_whole_estimate_base,
                (option.pix2pixsize, option.pix2pixsize),
                interpolation=cv2.INTER_CUBIC)

            # Merging the patch estimation into the base estimate using our merge network:
            # We feed the patch estimation and the same region from the updated base estimate to the merge network
            # to generate the target estimate for the corresponding region.
            pix2pix_model.set_input(patch_whole_estimate_base,
                                    patch_estimation)

            # Run merging network
            pix2pix_model.test()
            visuals = pix2pix_model.get_current_visuals()

            prediction_mapped = visuals['fake_B']
            prediction_mapped = (prediction_mapped + 1) / 2
            prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

            mapped = prediction_mapped

            # We use a simple linear polynomial to make sure the result of the merge network would match the values of
            # base estimate
            p_coef = np.polyfit(mapped.reshape(-1),
                                patch_whole_estimate_base.reshape(-1),
                                deg=1)
            merged = np.polyval(p_coef,
                                mapped.reshape(-1)).reshape(mapped.shape)

            merged = cv2.resize(merged, (org_size[1], org_size[0]),
                                interpolation=cv2.INTER_CUBIC)

            # Get patch size and location
            w1 = rect[0]
            h1 = rect[1]
            w2 = w1 + rect[2]
            h2 = h1 + rect[3]

            # To speed up the implementation, we only generate the Gaussian mask once with a sufficiently large size
            # and resize it to our needed size while merging the patches.
            if mask.shape != org_size:
                mask = cv2.resize(mask_org, (org_size[1], org_size[0]),
                                  interpolation=cv2.INTER_LINEAR)

            to_be_merged_to = image_and_patches.estimation_updated_image

            # Update the whole estimation:
            # We use a simple Gaussian mask to blend the merged patch region with the base estimate to ensure seamless
            # blending at the boundaries of the patch region.
            to_be_merged_to[h1:h2, w1:w2] = np.multiply(
                to_be_merged_to[h1:h2, w1:w2], 1 - mask) + np.multiply(
                    merged, mask)
            image_and_patches.set_updated_estimate(to_be_merged_to)

        # Output the result
        path = os.path.join(result_dir, f"{images.name}_boosting_depth")
        print("Output size {}".format(
            image_and_patches.estimation_updated_image.shape))
        midas_utils.write_depth(
            path,
            cv2.resize(image_and_patches.estimation_updated_image,
                       (input_resolution[1], input_resolution[0]),
                       interpolation=cv2.INTER_CUBIC),
            bits=2,
            colored=True)
        print("Time taken: {} seconds".format(time.time() - start_time))

    print("finished")


# Generating local patches to perform the local refinement described in section 6 of the main paper.
def generate_patches(img, base_size):
    # Compute the gradients as a proxy of the contextual cues.
    img_gray = rgb2gray(img)
    whole_grad = np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)) +\
        np.abs(cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3))

    threshold = whole_grad[whole_grad > 0].mean()
    whole_grad[whole_grad < threshold] = 0

    # We use the integral image to speed-up the evaluation of the amount of gradients for each patch.
    gf = whole_grad.sum() / len(whole_grad.reshape(-1))
    grad_integral_image = cv2.integral(whole_grad)

    # Variables are selected such that the initial patch size would be the receptive field size
    # and the stride is set to 1/3 of the receptive field size.
    blsize = int(round(base_size / 2))
    stride = int(round(blsize * 0.75))

    # Get initial Grid
    patch_bound_list = apply_grid_patch(blsize, stride, img, [0, 0, 0, 0])

    # Refine initial Grid of patches by discarding the flat (in terms of gradients of the rgb image) ones. Refine
    # each patch size to ensure that there will be enough depth cues for the network to generate a consistent depth map.
    print("Selecting patches ...")
    patch_bound_list = adaptive_selection(grad_integral_image,
                                          patch_bound_list, gf)

    # Sort the patch list to make sure the merging operation will be done with the correct order: starting from biggest
    # patch
    patch_set = sorted(patch_bound_list.items(),
                       key=lambda x: getitem(x[1], 'size'),
                       reverse=True)
    return patch_set


def adaptive_selection(integral_grad, patch_bound_list, gf):
    """
    Adaptively select patches

    :param integral_grad:
    :param patch_bound_list:
    :param gf:
    :return:
    """
    patch_list = {}
    count = 0
    height, width = integral_grad.shape

    search_step = int(32 / factor)

    # Go through all patches
    for c in range(len(patch_bound_list)):
        # Get patch
        bbox = patch_bound_list[str(c)]['rect']

        # Compute the amount of gradients present in the patch from the integral image.
        cgf = get_gradient_density_from_integral(integral_grad,
                                                 bbox) / (bbox[2] * bbox[3])

        # Check if patching is beneficial by comparing the gradient density of the patch to
        # the gradient density of the whole image
        if cgf >= gf:
            bbox_test = bbox.copy()
            patch_list[str(count)] = {}

            # Enlarge each patch until the gradient density of the patch is equal
            # to the whole image gradient density
            while True:

                bbox_test[0] = bbox_test[0] - int(search_step / 2)
                bbox_test[1] = bbox_test[1] - int(search_step / 2)

                bbox_test[2] = bbox_test[2] + search_step
                bbox_test[3] = bbox_test[3] + search_step

                # Check if we are still within the image
                if bbox_test[0] < 0 or bbox_test[1] < 0 or bbox_test[1] + bbox_test[3] >= height \
                        or bbox_test[0] + bbox_test[2] >= width:
                    break

                # Compare gradient density
                cgf = get_gradient_density_from_integral(
                    integral_grad, bbox_test) / (bbox_test[2] * bbox_test[3])
                if cgf < gf:
                    break
                bbox = bbox_test.copy()

            # Add patch to selected patches
            patch_list[str(count)]['rect'] = bbox
            patch_list[str(count)]['size'] = bbox[2]
            count = count + 1

    # Return selected patches
    return patch_list


def double_estimate(img, size1, size2, pix2pix_size):
    """
    Generate a double-input depth estimation

    :param img: input image
    :param size1: low resolution estimation size
    :param size2: high resolution estimation size
    :param pix2pix_size: size of pix2pix
    :return:
    """
    # Generate the low resolution estimation
    estimate1 = single_estimate(img, size1)
    # Resize to the inference size of merge network.
    estimate1 = cv2.resize(estimate1, (pix2pix_size, pix2pix_size),
                           interpolation=cv2.INTER_CUBIC)

    # Generate the high resolution estimation
    estimate2 = single_estimate(img, size2)
    # Resize to the inference size of merge network.
    estimate2 = cv2.resize(estimate2, (pix2pix_size, pix2pix_size),
                           interpolation=cv2.INTER_CUBIC)

    # Inference on the merge model
    pix2pix_model.set_input(estimate1, estimate2)
    pix2pix_model.test()
    visuals = pix2pix_model.get_current_visuals()
    prediction_mapped = visuals['fake_B']
    prediction_mapped = (prediction_mapped + 1) / 2
    prediction_mapped = (prediction_mapped - torch.min(prediction_mapped)) / (
        torch.max(prediction_mapped) - torch.min(prediction_mapped))
    prediction_mapped = prediction_mapped.squeeze().cpu().numpy()

    return prediction_mapped


def single_estimate(img, m_size):
    """
    Generate a single-input depth estimation

    :param img: input image array
    :param m_size: image size to be resized to so that it fits into GPU
    :return:
    """
    if m_size > GPU_threshold:
        print(" \t \t DEBUG| GPU THRESHOLD REACHED", m_size, '--->',
              GPU_threshold)
        m_size = GPU_threshold

    return estimate_midas(img, m_size)


# Inference on MiDas-v2
def estimate_midas(img, m_size):
    """
    MiDas -v2 forward pass script adapted from https://github.com/intel-isl/MiDaS/tree/v2

    :param img:
    :param m_size: image size to be resized to so that it fits into GPU
    :return: estimated depth map array
    """

    transform = Compose([
        Resize(
            m_size,
            m_size,
            resize_target=None,
            keep_aspect_ratio=True,
            ensure_multiple_of=32,
            resize_method="upper_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    img_input = transform({"image": img})["image"]

    # Forward pass
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        prediction = midas_model.forward(sample)

    prediction = prediction.squeeze().cpu().numpy()
    prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]),
                            interpolation=cv2.INTER_CUBIC)

    # Normalization
    depth_min = prediction.min()
    depth_max = prediction.max()

    if depth_max - depth_min > np.finfo("float").eps:
        prediction = (prediction - depth_min) / (depth_max - depth_min)
    else:
        prediction = 0

    return prediction


class DepthModel(BaseDepthModel):

    def __init__(self, name: str, args, parser):
        vars(args)['net_receptive_field_size'] = 384
        vars(args)['patch_netsize'] = 2 * 384
        vars(args)['pix2pixsize'] = 1024
        vars(args)['max_res'] = np.inf
        vars(args)['gpu_ids'] = "-1"
        super().__init__(name, args, parser)

    def get_depth_map(self):
        return boosting_get_depth_map(self.args, self.parser)

    @staticmethod
    def require_normalization():
        return True
