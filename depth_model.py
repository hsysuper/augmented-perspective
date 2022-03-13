import argparse
import contextlib
import os
import pathlib
import numpy as np

from models.monodepth2 import get_depth_map as get_monodepth_depth_map
from models.boosting import get_depth_map as get_boosting_depth_map


"""
USAGE:
From the augmented-perspective directory...

1) CHOOSE THE IMAGES TO WORK ON

    DEFAULT (all images in 'assets/'):
        python -m depth_model
    ENTIRE DIR:
        python -m depth_model --image_path assets/
    SINGLE FILE:
        python -m depth_model --image_path assets/test_image.jpg

2) CHOOSE THE MODELS TO TEST

    DEFAULT (all models):
        python -m depth_model
    MONODEPTH2 ONLY:
        python -m depth_model --monodepth2
    BOOSTING ONLY:
        python -m depth_model --boosting
"""


class DepthModel:
    def __init__(self, name: str, args, parser):
        self.name = name
        if os.path.isdir(args.image_path):
            vars(args)['image_path'] = '../../' + args.image_path
            vars(args)['image_files'] = None
        else:
            path = pathlib.Path(args.image_path)
            vars(args)['image_path'] = '../../' + str(path.parent)
            vars(args)['image_files'] = [str(path.name)]
        vars(args)['output_path'] = '../../' + args.output_path
        self.args = args
        self.parser = parser


class MonodepthDM(DepthModel):
    def __init__(self, name: str, args, parser):
        vars(args)['model_name'] = "mono_1024x320"
        super().__init__(name, args, parser)

    def get_depth_map(self):
        print(self.args)
        return get_monodepth_depth_map(self.args, self.parser)


class BoostingDM(DepthModel):
    def __init__(self, name: str, args, parser):
        vars(args)['net_receptive_field_size'] = 384
        vars(args)['patch_netsize'] = 2 * 384
        vars(args)['pix2pixsize'] = 1024
        vars(args)['max_res'] = np.inf
        vars(args)['gpu_ids'] = "-1"
        super().__init__(name, args, parser)

    def get_depth_map(self):
        return get_boosting_depth_map(self.args, self.parser)

"""
Context manager to easily set working directory when running
each depth estimation method so we don't need to refactor as much.
"""
@contextlib.contextmanager
def change_path(newdir):
    old_path = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try: yield
    finally: os.chdir(old_path)


"""
Arg parser for CMD line.
"""
def get_parser():
    parser = argparse.ArgumentParser(description='Depth model initialization.')
    parser.add_argument('--image_path', type=str, default='assets/',
                        help='path to a test image or folder of images')
    parser.add_argument('--monodepth2', action='store_true', default=False)
    parser.add_argument('--boosting', action='store_true', default=False)
    parser.add_argument('--output_path', type=str, default='outputs/')
    return parser


DEPTH_MODELS = [
    (MonodepthDM, 'monodepth2'),
    (BoostingDM, 'boosting'),
    # Add more models here...
]


def main():
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    # Run all the models by default if none are specified.
    if not args.monodepth2 and not args.boosting:
        vars(args)['monodepth2'] = True
        vars(args)['boosting'] = True

    for cls, name in DEPTH_MODELS:
        if not vars(args)[name]:
            continue

        args_cp = argparse.Namespace(**vars(args))
        parser_cp = argparse.ArgumentParser(parents=[parser], add_help=False)
        dm = cls(name, args_cp, parser_cp)
        with change_path(f'./models/{name}'):
            dm.get_depth_map()


if __name__ == '__main__':
    main()
