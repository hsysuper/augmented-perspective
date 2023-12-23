import pathlib


def get_intrinsic_matrix(image_path: pathlib.Path) -> pathlib.Path:
    intrinsic_matrices_map = {
        pathlib.Path("images/kitti1.png"): pathlib.Path("intrinsic_matrices/kitti_calib_cam_to_cam_09_26.txt"),
        pathlib.Path("images/kitti2.png"): pathlib.Path("intrinsic_matrices/kitti_calib_cam_to_cam_09_28.txt"),
        pathlib.Path("images/kitti3.png"): pathlib.Path("intrinsic_matrices/kitti_calib_cam_to_cam_09_26.txt"),
        pathlib.Path("images/kitti4.png"): pathlib.Path("intrinsic_matrices/kitti_calib_cam_to_cam_09_26.txt"),
        pathlib.Path("images/kitti5.png"): pathlib.Path("intrinsic_matrices/kitti_calib_cam_to_cam_09_26.txt"),
    }
    script_dir = pathlib.Path(__file__).resolve().parent
    if image_path.is_relative_to(script_dir):
        image_path = image_path.relative_to(script_dir)
    matrix_path = intrinsic_matrices_map.get(image_path)
    if matrix_path:
        return script_dir / matrix_path
    return None
