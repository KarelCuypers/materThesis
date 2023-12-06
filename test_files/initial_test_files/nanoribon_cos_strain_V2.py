import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt
from export_xyz import export_xyz


def sinusoidal_strain(c, k, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""

    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = 0
        uz = c * np.cos(k[0]*x + k[1]*y)
        return x + ux, y + uy, z + uz

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-beta * w)

    return displacement, strained_hopping


k = (1/2*pi)*np.array([1, 0])

model = pb.Model(
    graphene.monolayer(gamma3=False),
    pb.translational_symmetry(a1=True, a2=False),
    sinusoidal_strain(1, k),
    pb.rectangle(2)
)


model.plot()
model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm
plt.show()

solver = pb.solver.lapack(model)

a = graphene.a_cc * sqrt(3)  # ribbon unit cell length

bands = solver.calc_bands(-pi / a, pi / a)
bands.plot()
plt.show()

position = model.system.xyz

export_xyz("testxyz", position, model.lattice.vectors[0]*10, model.lattice.vectors[1]*10,
           np.array([0, 0, 100]), ['c']*position.shape[0])
