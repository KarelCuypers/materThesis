import pybinding as pb
import numpy as np


def find_ldos_test(name, repeat):
    load_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/lattice_files/'
    save_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/ldos_files/'

    lattice = pb.load(f'{load_path}lattice_{name}.pbz')

    model = pb.Model(lattice,
                     pb.primitive(repeat, repeat),
                     pb.translational_symmetry(a1=repeat*lattice.vectors[0][0], a2=repeat*lattice.vectors[1][1])
                     )

    model.plot()

    kpm = pb.kpm(model)

    ldos = kpm.calc_dos(energy=np.linspace(-2.7, 2.7, 500), broadening=0.01, num_random=100)

    pb.save(ldos, f'{save_path}ldos_{name}')
    print('ldos found')


name = f'supercell_size_10_test'
find_ldos_test(name, 2)
