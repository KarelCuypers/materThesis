import pybinding as pb
import numpy as np
from functions.bilayer_4atom import bilayer_4atom
from functions.uniform_strain import uniform_strain
from functions.export_xyz import export_xyz


def create_xyz(name, c_x, c_y):

    # x-direction is zz
    # y-direction is ac
    c0 = 0.335

    a1, a2 = bilayer_4atom().vectors[0] * (1 + c_x), bilayer_4atom().vectors[1] * (1 + c_y)

    times_l1 = 20
    times_l2 = 20

    strained_model = pb.Model(
        bilayer_4atom(),
        pb.primitive(times_l1, times_l2),
        uniform_strain(c_x, c_y),
    )

    position = strained_model.system.xyz

    path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/xyz_files/'

    export_xyz(f'{path}{name}', position, times_l1 * a1, times_l1 * a2,
               np.array([0, 0, -c0]), ['A1'] * position.shape[0]
               )

    print("Done")
