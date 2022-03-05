import argparse
import contextlib
import os

from models.monodepth2 import get_depth_map as get_monodepth_depth_map


"""
USAGE:
From the augmented-perspective directory...
    python -m depth_model --image_path assets/test_image.jpg
"""

class DepthModel:
    def __init__(self, name: str, args):
        self.name = name
        vars(args)['image_path'] = '../../' + args.image_path
        self.args = args


class MonodepthDM(DepthModel):
    def __init__(self, name: str, args):
        # Pre-trained model options: ["mono_640x192", "mono_1024x320"]
        vars(args)['model_name'] = "mono_1024x320"
        super().__init__(name, args)

    def get_depth_map(self):
        return get_monodepth_depth_map(self.args)


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
def parse_args():
    parser = argparse.ArgumentParser(description='Depth model initialization.')
    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images',
                        required=True)
    return parser.parse_args()


DEPTH_MODELS = [
    (MonodepthDM, 'monodepth2'),
    # Add more models here...
]


if __name__ == '__main__':
    args = parse_args()
    dms = [(cls(name, args), name) for cls, name in DEPTH_MODELS]
    for dm, name in dms:
        with change_path(f'./models/{name}'):
            depth_map = dm.get_depth_map()
            if depth_map:
                print(depth_map.shape)
                print(depth_map)
