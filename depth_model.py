import argparse
import contextlib
import importlib
import os
import pathlib
import sys

import models.base_model
"""
USAGE:
From the augmented-perspective directory...

1) CHOOSE THE IMAGES TO WORK ON

    ENTIRE DIR:
        python -m depth_model --image_path datasets/
    SINGLE FILE:
        python -m depth_model --image_path <path-to-image>

2) CHOOSE THE MODELS TO TEST

    DEFAULT (all models):
        python -m depth_model
    A SINGLE MODEL ONLY:
        python -m depth_model --depth_model <model-name>
"""


@contextlib.contextmanager
def change_path(new_dir: pathlib.Path):
    """
    Context manager to easily set working directory when running each depth estimation method, so we don't need to
    refactor as much.

    :param new_dir: new directory to work in
    :return: None
    """
    old_path = pathlib.Path.cwd()
    os.chdir(new_dir.resolve())
    try:
        yield
    finally:
        os.chdir(old_path)


def get_depth_model_list():
    """
    :return: A list of depth models that can be imported
    """
    script_dir = pathlib.Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_list = [f.parts[-1] for f in models_dir.iterdir() if f.is_dir()]
    return models_list


def get_parser():
    """
    Arg parser for command line.
    """
    models_list = get_depth_model_list()
    parser = argparse.ArgumentParser(description='Depth model initialization.')
    parser.add_argument(
        "--depth_model",
        type=str,
        help=
        f"name of depth model used for creating depth map under models/, allowed = {models_list}",
        choices=models_list)
    parser.add_argument('--image_path',
                        type=pathlib.Path,
                        default='assets/',
                        help='path to a test image or folder of images')
    parser.add_argument("--output_path",
                        type=pathlib.Path,
                        help="output path",
                        default=pathlib.Path("outputs"))
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
        depth_model_module = importlib.import_module(
            f"models.{model_name}.depth_prediction")
        depth_model_class: models.base_model.BaseDepthModel.__class__ = depth_model_module.DepthModel

        args_cp = argparse.Namespace(**vars(args))
        parser_cp = argparse.ArgumentParser(parents=[parser], add_help=False)
        depth_model = depth_model_class(model_name, args_cp, parser_cp)
        with change_path(
                pathlib.Path(__file__).resolve().parent /
                f'models/{model_name}'):
            depth_model.get_depth_map()


if __name__ == '__main__':
    run_depth_model(sys.argv)
