import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt


def y_sinusoidal_strain(c, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""

    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = 0
        uz = c * np.cos(y)
        return x + ux, y + uy, z + uz

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-beta * w)

    return displacement, strained_hopping


def x_sinusoidal_strain(c, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""

    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = 0
        uz = c * np.cos(x)
        return x + ux, y + uy, z + uz

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-beta * w)

    return displacement, strained_hopping


armchair_model = pb.Model(
    graphene.monolayer(gamma3=False),
    pb.translational_symmetry(a1=True, a2=False),
    x_sinusoidal_strain(2),
    pb.rectangle(2)
)

zigzag_model = pb.Model(
    graphene.monolayer(gamma3=False),
    pb.translational_symmetry(a1=False, a2=True),
    y_sinusoidal_strain(2),
    pb.rectangle(2)
)

armchair_model.plot()
armchair_model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm
plt.show()

zigzag_model.plot()
zigzag_model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm
plt.show()

arm_solver = pb.solver.lapack(armchair_model)
zig_solver = pb.solver.lapack(zigzag_model)

a = graphene.a_cc * sqrt(3)  # ribbon unit cell length
d = graphene.a_cc * 3

bands = arm_solver.calc_bands(-pi / a, pi / a)
bands.plot()
plt.show()

bands = arm_solver.calc_bands(0, 4*pi / a)
bands.plot()
plt.show()

bands = zig_solver.calc_bands(-pi / d, pi / d)
bands.plot()
plt.show()

bands = zig_solver.calc_bands(0, 4*pi / d)
bands.plot()
plt.show()

"""a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [4*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = arm_solver.calc_bands(K1, Gamma, M, K2)
bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
plt.show()"""
