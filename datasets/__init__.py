import importlib
import logging
import pathlib


def get_intrinsic_matrix(image_path: pathlib.Path) -> pathlib.Path:
    """
    Get intrinsic matrix of image in this directory

    image_path: full path to image
    return: None if no intrinsic matrix can be found
    """
    image_path = image_path.resolve()
    script_dir = pathlib.Path(__file__).resolve().parent
    dataset_dirs = [f for f in script_dir.iterdir() if f.is_dir()]
    logging.debug(
        f"list of datasets: {dataset_dirs}, current image: {image_path}")
    for dataset_dir in dataset_dirs:
        if not image_path.is_relative_to(dataset_dir):
            logging.debug(
                f"{image_path} is not relative to {dataset_dir}, next one")
            continue
        logging.debug(
            f"{image_path} is relative to {dataset_dir}, trying to get intrinsic matrix"
        )
        try:
            dataset = importlib.import_module(
                f"{script_dir.stem}.{dataset_dir.stem}")
            return dataset.get_intrinsic_matrix(image_path)
        except ImportError as e:
            logging.warning(f"Failed to import dataset module: {e}")
        break
    return None
