"""Strain a triangular system by pulling on its vertices"""
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi

pb.pltutils.use_style()


def circle(radius):
    def contains(x, y, z):
        return np.sqrt(x**2 + y**2) < radius
    return pb.FreeformShape(contains, width=[2*radius, 2*radius])


def triaxial_strain(c):
    """Strain-induced displacement and hopping energy modification"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 2*c * x*y
        uy = c * (x**2 - y**2)
        return x + ux, y + uy, z

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-3.37 * w)

    return displacement, strained_hopping


# graphene disc with a diameter of 100nm and a strain c = 10% / D_m same size as in the paper
model = pb.Model(
    graphene.monolayer(),
    circle(30),
    triaxial_strain(c=0.1/30)
)

model.plot()
plt.show()

kpm = pb.kpm(model)

# local density of states for the A and B sublattice
for sub_name in ['A', 'B']:
    ldos = kpm.calc_ldos(energy=np.linspace(-1, 1, 500), broadening=0.03,
                         position=[0, 0], sublattice=sub_name)
    ldos.plot(label=sub_name, ls="--" if sub_name == "B" else "-")
pb.pltutils.legend()
plt.show()

# effect of strain is clear split in Landau levels

# total density of states
dos = kpm.calc_dos(energy=np.linspace(-1, 1, 200), broadening=0.05, num_random=16)
dos.plot()
plt.show()
