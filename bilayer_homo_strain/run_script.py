import matplotlib.pyplot as plt
from functions.create_xyz import *
from functions.create_lattice import create_lattice
from functions.find_bands_test import find_bands_test
#from functions.find_ldos_test import find_ldos_test

size = [50]

c_x = 0.03
c_y = 0

for i in size:
    name = f'supercell_size_{i}_{c_x}_strain'
    # x-direction is zz
    # y-direction is ac
    c0 = 0.335

    a1, a2 = bilayer_4atom().vectors[0] * (1 + c_x), bilayer_4atom().vectors[1] * (1 + c_y)

    times_l1 = i
    times_l2 = i

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

    create_lattice(name)
    #find_bands_test(name, 5)
    #find_ldos_test(name, 5)
