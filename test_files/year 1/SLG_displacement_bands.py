import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt


def uniaxial_strain(c, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""

    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = c * y
        return x + ux, y + uy, z

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-beta * w)
    return displacement, strained_hopping


def strained_sgl_lattice(c_x, c_y):
    a = 0.24595  # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    t = -2.8  # [eV] nearest neighbour hopping

    lat = pb.Lattice(a1=[(1 + c_x) * a, 0],
                     a2=[(1 + c_x) * a / 2, (1 + c_y) * a / 2 * sqrt(3)])
    lat.add_sublattices(('A', [0, -(1 + c_y) * a_cc / 2]),
                        ('B', [0, (1 + c_y) * a_cc / 2]))
    lat.add_hoppings(
        # inside the main cell
        ([0, 0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lat


def square_lattice(d, t):
    lat = pb.Lattice(a1=[d, 0], a2=[0, d])
    lat.add_sublattices(('A', [0, 0]))
    lat.add_hoppings(([0, 1], 'A', 'A', t),
                     ([1, 0], 'A', 'A', t))
    return lat


strain_model = pb.Model(strained_sgl_lattice(0, 0.12),  # new deformed lattice
                        uniaxial_strain(0),  # still included for hopping change doesn't include deformation
                        pb.translational_symmetry()
                        )
strain_model.lattice.plot()
plt.show()

point_list = strain_model.lattice.brillouin_zone()
print(point_list)

vectors = strain_model.lattice.reciprocal_vectors()
print(vectors)

b1 = vectors[0]
b2 = vectors[1]

strain_model.lattice.plot_brillouin_zone()
plt.show()

solver = pb.solver.lapack(strain_model)

a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4 * pi / (3 * np.sqrt(3) * a_cc), 0]
S = [1/2*(b1[0] + b2[0]), 1/2*(b1[1] + b2[1])]
R = point_list[2]

bands = solver.calc_bands(Gamma, R, S)

strain_model.lattice.plot_brillouin_zone()
bands.plot_kpath(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()
bands.plot(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()

