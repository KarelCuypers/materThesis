import numpy as np
import pybinding as pb


def unit_cell(l1, l2):
    """Make the shape of a unit cell

    Parameters
    ----------
    l1, l2 : np.ndarray, np.ndarray
        Unit cell vectors.
    """

    A = np.array([0, 0])
    B = np.array([l1[0], l1[1]])
    C = np.array([l1[0] + l2[0], l1[1] + l2[1]])
    D = np.array([l2[0], l2[1]])

    return pb.Polygon([A, B, C, D])
