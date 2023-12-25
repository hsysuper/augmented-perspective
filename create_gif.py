import argparse
import io
import pathlib
import sys

from PIL import Image
import reverse_argparse

from augmented_perspective import run_augmented_perspective, get_argument_parser
"""
Usage:

Option 1: Generate augmented perspectives using this tool before creating GIFs

    python -m create_gif --image_path <path-to-image> --depth_map_path <depth-map> --depth_model <model-name> \
             --loop_reverse --output_path <output_path>

Option 2: Use existing augmented perspectives to generate GIFs

Existing augmented perspectives must from a single directory and the images will be ordered using sorted() function
on its path string. In this mode, the script will look for existing images in [output_path] 

    python -m create_gif --use_existing_image --loop_reverse --output_path <output_path>

"""


def run_create_gif(argv):
    sys.argv = argv
    parser = argparse.ArgumentParser(
        parents=[get_argument_parser()],
        add_help=False,
        description="Tool to create GIF clips from augmented perspective images"
    )
    parser.add_argument(
        "--use_existing_image",
        type=bool,
        help="Try to use existing image from output_path to create GIF",
        action=argparse.BooleanOptionalAction,
        default=False)
    parser.add_argument(
        "--loop_reverse",
        type=bool,
        help="Play images in reverse order after playing them forward",
        action=argparse.BooleanOptionalAction,
        default=False)
    parser.set_defaults(angle=20, translation=-0.5, frames=30)
    args = parser.parse_args()

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

    if not files:
        print(f"No files found in {args.output_path}", file=sys.stderr)
        sys.exit(1)

    frame_indices = [
        int(file[file.rfind('_') + 1:file.rfind('.')]) for file in files
    ]
    files = [x for _, x in sorted(zip(frame_indices, files))]

    images = [Image.open(p) for p in files]

    if args.loop_reverse:
        images += [Image.open(p) for p in files[-2::-1]]

    temp_buffer = io.BytesIO()
    images[0].save(temp_buffer,
                   format="gif",
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)
    output_gif_path: pathlib.Path = args.output_path / f"{args.output_path.stem}.gif"
    output_gif_path.unlink(missing_ok=True)
    output_gif_path.write_bytes(temp_buffer.getbuffer().tobytes())


if __name__ == '__main__':
    run_create_gif(sys.argv)
