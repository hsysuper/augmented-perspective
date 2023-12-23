import pathlib
import sys

from PIL import Image

from augmented_perspective import run_augmented_perspective, parse_args
"""
Usage:
    python -m create_gif --image_path <path-to-image> --depth_model <model-name>
"""

if __name__ == '__main__':
    # TODO: Manually adjust these 3 parameters before running the script.
    ANGLE = 20
    TRANSLATION = -0.5
    FRAMES = 30

    args = parse_args()

    image_name = pathlib.Path(args.image_path).stem
    output_name = f"{image_name}_{args.depth_model}"
    dirname = f"{output_name}_{ANGLE},{TRANSLATION}"
    folder = pathlib.Path(f"output_images/{dirname}")

    # Generate all frames of images.
    run_augmented_perspective(sys.argv,
                              save_filled_only=True,
                              ANGLE=ANGLE,
                              TRANSLATION=TRANSLATION,
                              FRAMES=FRAMES,
                              output_directory=folder)

    # Create gif once all images have been generated.
    files = list(pathlib.Path(folder).glob('*.png'))
    files = list(map(str, files))

    frame_idxs = [
        int(file[file.rfind('_') + 1:file.rfind('.')]) for file in files
    ]
    files = [x for _, x in sorted(zip(frame_idxs, files))]

    images = [Image.open(p) for p in files]

    images[0].save(folder / f"{dirname}.gif",
                   save_all=True,
                   append_images=images[1:],
                   duration=100,
                   loop=0)
