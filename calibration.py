import argparse
import numpy as np


def parse_args():
    """
    Arg parser for CMD line.
    """
    parser = argparse.ArgumentParser(description='Calibration module initialization.')
    parser.add_argument('--depth_map_path', type=str,
                        help='path to a test image depth map',
                        required=True)
    return parser.parse_args()


def calibrate(depth_map):
    """
    Return the camera matrix used to take the input image

      -------------> u
      |\
      | \
      |  \
      |   \
      |v   \ depth

    Assuming same axis in camera and real world axis, thus
    x = u, y = v, z = depth

    :param depth_map: depth map of the image (H, W)
    :return: Camera matrix (3, 4)
    """

    def get_u_row(x, y, z, u, v):
        return np.array([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u], dtype=np.float64)

    def get_v_row(x, y, z, u, v):
        return np.array([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v], dtype=np.float64)

    print("depth_map.shape", depth_map.shape)
    H, W = depth_map.shape
    p_matrix_list = []
    stride = 4
    for u in range(0, W, stride):
        for v in range(0, H, stride):
            p_matrix_list.append(get_u_row(u, v, depth_map[v, u], u, v))
            p_matrix_list.append(get_v_row(u, v, depth_map[v, u], u, v))
    p_matrix = np.zeros((len(p_matrix_list), 12), dtype=np.float64)
    for i in range(len(p_matrix_list)):
        p_matrix[i] = p_matrix_list[i]
    print("SVD on p_matrix", p_matrix.shape)
    u, s, vh = np.linalg.svd(p_matrix, full_matrices=True)
    m_values = vh[-1]
    print("m_values\n", m_values)
    M = m_values.reshape(3, 4)
    print("M\n", M)
    M_calibrated = M / M[2, 3]
    print("M_calibrated\n", M_calibrated)
    return M_calibrated


if __name__ == '__main__':
    args = parse_args()
    depth_map_path = args.depth_map_path

    depth_map = np.load(depth_map_path)
    M = calibrate(depth_map)
    print("Final M:\n", M)
