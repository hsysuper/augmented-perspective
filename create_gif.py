import argparse
import pathlib
import sys

from PIL import Image
import reverse_argparse

from augmented_perspective import run_augmented_perspective, get_argument_parser
"""
Usage:
    python -m create_gif --image_path <path-to-image> --depth_map_path <depth-map> \
            --depth_model <model-name> --output_path <output_path>
"""

if __name__ == '__main__':
    parser = get_argument_parser()
    parser.add_argument(
        "--use_existing_image",
        type=bool,
        help="Try to use existing image from output_path to create GIF",
        action=argparse.BooleanOptionalAction,
        default=False)
    parser.set_defaults(angle=20, translation=-0.5, frames=30)
    args = parser.parse_args(sys.argv)

    if not args.use_existing_image:
        # Get another instance of the default parser so that the unparser can produce the correct command line arguments
        default_parser = get_argument_parser()
        unparser = reverse_argparse.ReverseArgumentParser(default_parser, args)
        argv = unparser.get_effective_command_line_invocation().split()
        # Generate all frames of images.
        run_augmented_perspective(argv)

    # Create gif once all images have been generated.
    files = list(pathlib.Path(args.output_path).glob('*.png'))
    files = list(map(str, files))

    frame_indices = [
        int(file[file.rfind('_') + 1:file.rfind('.')]) for file in files
    ]
    files = [x for _, x in sorted(zip(frame_indices, files))]

    images = [Image.open(p) for p in files]

    images[0].save(args.output_path / f"{args.output_path.stem}.gif",
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)
