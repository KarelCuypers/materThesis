import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt

pb.pltutils.use_style()


def uniaxial_strain(c, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = c*y
        return x + ux, y + uy, z

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w_intra = l / graphene.a_cc - 1
        w_inter = l / 0.335 - 1

        if np.array_equiv(z1, z2):
            return energy * np.exp(-beta * w_intra)
        else:
            return 0.48 * np.exp(-beta * w_inter)

    return displacement, strained_hopping


model = pb.Model(graphene.bilayer(gamma3=False),
                 pb.translational_symmetry(),
                 uniaxial_strain(0)
                 )
model.plot()
plt.show()

solver = pb.solver.lapack(model)

point_list = model.lattice.brillouin_zone()
print(point_list)

vectors = model.lattice.reciprocal_vectors()
print(vectors)

b1 = vectors[0]
b2 = vectors[1]

a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*np.sqrt(3)*a_cc), 0]
S = [0, b1[1]]
R = point_list[4]


bands = solver.calc_bands(Gamma, R, S, step=0.001)
model.lattice.plot_brillouin_zone()
bands.plot_kpath(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()
bands.plot(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()

'''bands = solver.calc_bands(K1, Gamma, S, R)
bands.plot(point_labels=['K', r'$\Gamma$', 'S', 'R'])
plt.show()'''
