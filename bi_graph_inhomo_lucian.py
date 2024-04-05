import pybinding as pb
from pybinding.repository import graphene
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt
from functions.contour_dos import contour_dos
from functions.calculate_surfaces import calculate_surfaces
from functions.draw_contour import draw_contour
from functions.export_xyz import export_xyz
from functions.create_lattice import create_lattice
from make_lattice_bigrhbn import add_one_atomic_layer, unit_cell, a, a_cc, t, c0


def strained_lattice(nearest_neighbors=1, onsite=(0, 0), strain=[0, 0], **kwargs):

    lat = pb.Lattice(a1=[a * (1+strain[0]), 0],
                     a2=[a/2, a/2 * sqrt(3)]
                     )

    t_nn = graphene.t_nn
    # The next-nearest hoppings shift the Dirac point away from zero energy.
    # This will push it back to zero for consistency with the first-nearest model.
    onsite_offset = 0 if nearest_neighbors < 2 else 3 * kwargs.get('t_nn', t_nn)

    lat.add_sublattices(
        ('A', [a/4 * (1+strain[0]), +a_cc/4], onsite[0] + onsite_offset),
        ('B', [-a/4 * (1+strain[0]),  -a_cc/4], onsite[1] + onsite_offset)
    )

    lat.register_hopping_energies({
        't': kwargs.get('t', t),
        't_nn': kwargs.get('t_nn', t_nn),
        't_nnn': kwargs.get('t_nnn', 0.05),
    })

    lat.add_hoppings(
        ([0,  0], 'A', 'B', 't'),
        ([1, -1], 'A', 'B', 't'),
        ([0, -1], 'A', 'B', 't')
    )

    if nearest_neighbors >= 2:
        lat.add_hoppings(
            ([0, -1], 'A', 'A', 't_nn'),
            ([0, -1], 'B', 'B', 't_nn'),
            ([1, -1], 'A', 'A', 't_nn'),
            ([1, -1], 'B', 'B', 't_nn'),
            ([1,  0], 'A', 'A', 't_nn'),
            ([1,  0], 'B', 'B', 't_nn'),
        )

    if nearest_neighbors >= 3:
        lat.add_hoppings(
            [( 1, -2), 'A', 'B', 't_nnn'],
            [( 1,  0), 'A', 'B', 't_nnn'],
            [(-1,  0), 'A', 'B', 't_nnn'],
        )

    if nearest_neighbors >= 4:
        raise RuntimeError("No more")

    lat.min_neighbors = 2
    return lat


def calculate_hoppings():
    """Produce both the displacement and hopping energy modifier"""
    @pb.hopping_energy_modifier
    def strained_hopping(x1, y1, z1, x2, y2, z2):
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_0 = 0.184*a
        v_pi = t * np.exp(-(r - a/sqrt(3)) / r_0)
        v_sig = - 0.48 * np.exp(-(r - 0.335) / r_0)
        dz = z1 - z2

        return v_pi * (1 - (dz/r)**2) + v_sig * (dz/r)**2

    return strained_hopping


n = 10
c_x = 1/(n-1)
print(c_x)
c_y = 1/(n-1)
strain = [c_x, c_y]

# make a graphene lattice and extract unit cell vectors a1 and a2
lattice_gr = strained_lattice(strain=strain)
a1_strained, a2_strained = lattice_gr.vectors[0], lattice_gr.vectors[1]
a1, a2 = strained_lattice().vectors[0], strained_lattice().vectors[1]

# make a unit cell shape from vectors l1 and l2

#q=2*n
q = 1

#strained_shape = unit_cell((n-1)*a1_strained, q*a2_strained)
#shape = unit_cell((n+1)*a1, q*a2)

strained_shape = pb.primitive((n-1), q)
shape = pb.primitive(n, q)

strained_model = pb.Model(
    lattice_gr,
    strained_shape,
    #add_one_atomic_layer(name='A2', position=[0, -a_cc/2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
    add_one_atomic_layer(name='A2', position=[a/4, +a_cc/4, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
    #add_one_atomic_layer(name='B2', position=[0, a_cc/2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
    add_one_atomic_layer(name='B2', position=[-a/4,  -a_cc/4, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
    #calculate_hoppings()
)

name = "hetro_coord"
position = strained_model.system.xyz
export_xyz(name, position, a1*n, a2*q, np.array([0, 0, 1]), ['A'] * position.shape[0])

'''m = 2

complete_lattice = create_lattice(name)
new_model = pb.Model(complete_lattice,
                     pb.primitive(m, m),
                     pb.translational_symmetry(a1=m*complete_lattice.vectors[0][0], a2=m*complete_lattice.vectors[1][1])
                     )
new_model.plot()'''

'''solver = pb.solver.lapack(new_model)
kpm = pb.kpm(new_model)

hbar = 4.136*10**(-15) #eV*s
t_0 = 2.7 #eV
a = 1.42 * 0.1 #nm
v_F = 3/2 * t_0 * a/hbar

Gnorm = 2*np.pi/(n*np.linalg.norm(a1))

#E = hbar * v_F * Gnorm/2
E = 2*np.pi*hbar*v_F/(sqrt(3)*n*np.linalg.norm(a1))
print(E)

# dos fig around the K1 point

energies = np.linspace(-2, 2, 2000)
dos = kpm.calc_dos(energy=energies, broadening=0.05)
plt.figure()
dos.plot()
plt.axvline(x=E, color='r', label='Dip')
plt.title(f'{n-1} over {n} strain dos')
plt.show()'''
