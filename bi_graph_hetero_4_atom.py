import pybinding as pb
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from pybinding.repository.graphene import a, a_cc, t
import numpy as np
from numpy import sqrt
from functions.export_xyz import export_xyz
from functions.create_lattice import create_lattice


def add_one_atomic_layer(name, position, onsite_energy, a1, a2, shape):
    """Add a layer made from single sublattice

    Parameters
    ----------

    name : str
        User friendly name for the sublattice.
    position : list
        Position in xyz coordinates.
    onsite_energy : float
        Onsite energy terms at the sublattice.
    a1, a2 : Tuple[float, float, float]
        Unit cell vectors.
    shape:
        Shape of the structure (unit cell).

    Returns
    -------

    Lattice site positions. Named tuple with x, y, z fields, each a 1D array.
    """

    @pb.site_generator(name, onsite_energy)
    def define_layer():
        lat = pb.Lattice(a1=a1, a2=a2)

        lat.add_sublattices((name, position, onsite_energy))

        model = pb.Model(
            lat,
            shape,
        )

        return model.system.positions

    return define_layer


def strained_lattice(onsite=(0, 0, 0, 0), strain=[0, 0]):
    lat = pb.Lattice(a1=[a * (1+strain[0]), 0],
                     a2=[0, 3*a_cc * (1+strain[1])])

    lat.add_sublattices(
        # layer 1
        ('A1', [0,  -a_cc/2,   0], onsite[0]),
        ('B1', [0,   a_cc/2,   0], onsite[1]),
    )

    lat.add_aliases(
        # Layer 1
        ('A3', 'A1', [a / 2 * (1+strain[0]), -a_cc/2 + 3*a_cc/2, 0]),
        ('B3', 'B1', [a / 2 * (1+strain[0]), a_cc/2 + 3*a_cc/2 , 0]),
        # layer 2
    )
    return lat

#n_list = [70, 75, 80, 85, 90, 95, 100]
n_list = [10]
m = 1

for n in n_list:

    name = f'supercell_size_{n}x{m}_a1_hetro_strain'

    c_x = 1 / (n - 1)
    print(c_x * 100)
    c_y = 0
    strain = [c_x, c_y]

    # make a graphene lattice and extract unit cell vectors a1 and a2
    lattice_gr = strained_lattice(strain=strain)
    a1_strained, a2_strained = lattice_gr.vectors[0], lattice_gr.vectors[1]
    a1, a2 = strained_lattice().vectors[0], strained_lattice().vectors[1]

    # make a unit cell shape from vectors l1 and l2

    q = 2 * n

    strained_shape = pb.primitive((n - 1), q)
    shape = pb.primitive(n, q)

    c0 = 0.335  # [nm] interlayer spacing
    strained_model = pb.Model(
        lattice_gr,
        strained_shape,
        add_one_atomic_layer(name='A2', position=[0, -a_cc/2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        add_one_atomic_layer(name='B2', position=[0, a_cc/2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        add_one_atomic_layer(name='A3', position=[a / 2, -a_cc / 2 + 3 * a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        add_one_atomic_layer(name='B3', position=[a / 2, a_cc / 2 + 3 * a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
    )

    xyz_name = f"hetro_{n}x{m}_coord"
    position = strained_model.system.xyz
    export_xyz(xyz_name, position, a1*n, a2*q, np.array([0, 0, 1]), ['A'] * position.shape[0])

    complete_lattice = create_lattice(xyz_name)
    new_model = pb.Model(complete_lattice,
                         pb.primitive(m, m),
                         pb.translational_symmetry(a1=m*complete_lattice.vectors[0][0], a2=m*complete_lattice.vectors[1][1])
                         )

    new_model.plot()
    kpm = pb.kpm(new_model)
    dos = kpm.calc_dos(energy=np.linspace(-5, 5, 10000), broadening=0.01, num_random=4*64)

    #pb.save(dos, f'ldos_{name}.pbz')

    #dos = pb.load(f'ldos_{name}.pbz')

    #dos.data = dos.data / max(dos.data)

    hbar = 4.136 * 10 ** (-15)  # eV*s
    t_0 = 2.7  # eV
    a = 1.42 * 0.1  # nm
    v_F = 3 / 2 * t_0 * a / hbar

    # Gnorm = 2*np.pi/(n*np.linalg.norm(a1))

    # E = hbar * v_F * Gnorm/2
    E = 2 * np.pi * hbar * v_F / (sqrt(3) * n * np.linalg.norm(a1_strained))
    print(E)

    # dos fig around the K1 point

    plt.figure()
    dos.plot()
    plt.axvline(x=E, color='r', label='Dip')
    plt.title(f'{n - 1} over {n} strain dos')
    # plt.xlim(-0.5, 0.5)
    # plt.ylim(0, 0.125)
    plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/hetero_dos/full_hetero_{n}x{m}_dos_normalised.png')
    plt.show()

