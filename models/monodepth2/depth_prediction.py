# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import glob
import os
import time

import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
from torchvision import transforms

from models.base_model import BaseDepthModel
from . import networks
from .utils import download_model_if_doesnt_exist


def disp_to_depth(disp, min_depth, max_depth):
    """
    Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.

    :param disp: inverse depth estimation scores from sigmoid output
    :param min_depth: minimum depth value
    :param max_depth: maximum depth value
    :return: converted depth value
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def monodepth2_get_depth_map(args):
    """
    Function to predict for a single image or folder of images

    :param args: arguments parsed in upper layer
    :return: arrry of depth estimation
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    device = torch.device(args.device)

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {
        k: v
        for k, v in loaded_dict_enc.items() if k in encoder.state_dict()
    }
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc,
                                          scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    print("Reading images from {}".format(args.image_path))
    if args.image_files is not None:
        # Only testing on a single image
        paths = []
        for image_file in args.image_files:
            paths.append(os.path.join(args.image_path, image_file))
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*'))
    else:
        raise Exception("Can not find args.image_path: {}".format(
            args.image_path))

    output_directory = args.output_path

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):
            start_time = time.time()

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height),
                                             pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            print("input_image.shape: ", input_image.shape)
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            print("disp.shape: ", disp.shape)
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width),
                mode="bilinear",
                align_corners=False)
            print("disp_resized.shape: ", disp_resized.shape)

            # Saving numpy file
            output_name = os.path.splitext(
                os.path.basename(image_path))[0] + "_monodepth2"
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            print("scaled_disp.shape: ", scaled_disp.shape)
            name_dest_npy = os.path.join(output_directory,
                                         "{}_disp.npy".format(output_name))
            np.save(name_dest_npy, scaled_disp.cpu().numpy())

            # Save file for calibration
            name_resized_dest_npy = os.path.join(
                output_directory, "{}_resized_disp.npy".format(output_name))
            np.save(name_resized_dest_npy, disp_resized.cpu().numpy())
            resized_disp_scaled, depth_resized = disp_to_depth(
                disp_resized, 1, 1000)
            name_resized_depth_npy = os.path.join(
                output_directory, "{}_depth.npy".format(output_name))
            disp_resized_np = depth_resized.squeeze().cpu().numpy()
            np.save(name_resized_depth_npy, disp_resized_np)
            print(disp_resized_np)

            # Saving colormapped depth image
            print("disp_resized_np.shape: ", disp_resized_np.shape)
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(),
                                              vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] *
                              255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory,
                                        "{}_depth.png".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".
                  format(idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))
            print("   - {}".format(name_dest_npy))
            print("Time taken: {} seconds".format(time.time() - start_time))

    print('-> Done!')


class DepthModel(BaseDepthModel):

    def __init__(self, name: str, args, parser):
        vars(args)['model_name'] = "mono_1024x320"
        super().__init__(name, args, parser)

    def get_depth_map(self):
        print(self.args)
        return monodepth2_get_depth_map(self.args)
