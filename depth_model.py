import argparse
import contextlib
import importlib
import os
import pathlib
import sys


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
        python -m depth_model --depth_model monodepth2
    BOOSTING ONLY:
        python -m depth_model --depth_model boosting
"""


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


def get_depth_model_list():
    script_dir = pathlib.Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_list = [f.parts[-1] for f in models_dir.iterdir() if f.is_dir()]
    return models_list


def get_parser():
    """
    Arg parser for CMD line.
    """
    models_list = get_depth_model_list()
    parser = argparse.ArgumentParser(description='Depth model initialization.')
    parser.add_argument("--depth_model", type=str,
                        help=f"name of depth model used for creating depth map under models/, allowed = {models_list}",
                        choices=models_list)
    parser.add_argument('--image_path', type=pathlib.Path, default='assets/',
                        help='path to a test image or folder of images')
    parser.add_argument("--output_path", type=pathlib.Path, help="output path", default=pathlib.Path("outputs"))
    parser.add_argument('--device', type=str, default='cpu')
    return parser


def run_depth_model(argv):
    print("before", sys.argv)
    sys.argv = argv
    print("after", sys.argv)
    parser = get_parser()
    args = parser.parse_args()
    args.output_path.mkdir(parents=True, exist_ok=True)

    models_list = get_depth_model_list()

    if args.depth_model:
        models_list = [args.depth_model]

    for model_name in models_list:
        depth_model_module = importlib.import_module(f"models.{model_name}.depth_prediction")
        depth_model = depth_model_module.DepthModel

        args_cp = argparse.Namespace(**vars(args))
        parser_cp = argparse.ArgumentParser(parents=[parser], add_help=False)
        dm = depth_model(model_name, args_cp, parser_cp)
        with change_path(f'./models/{model_name}'):
            dm.get_depth_map()


if __name__ == '__main__':
    run_depth_model(sys.argv)
