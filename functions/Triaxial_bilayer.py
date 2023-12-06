import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt

pb.pltutils.use_style()


def triaxial_strain(c, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 2*c * x*y
        uy = c * (x**2 - y**2)
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


size = 12
B_ps = 1000

a_cc = graphene.a_cc * 10**(-9)
h = 4*10**(-15)
beta = 3.37
c = a_cc*B_ps/(4*h*beta) * 10**-9
print(c)

IB = sqrt(h/B_ps)*10**9
print(IB)

unstrained_model = pb.Model(
    graphene.bilayer(),
    pb.regular_polygon(num_sides=6, radius=size, angle=pi)
)

strained_model = pb.Model(
    graphene.bilayer(),
    pb.regular_polygon(num_sides=6, radius=size, angle=pi),
    triaxial_strain(c=c)
)

fig, (ax1, ax2) = plt.subplots(2)
unstrained_model.plot(ax=ax1)
strained_model.plot(ax=ax2)
plt.show()

'''plt.figure()
kpm = pb.kpm(unstrained_model)
for sub_name in ['A1', 'B1', 'A2', 'B2']:
    ldos = kpm.calc_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03, position=[0, 0],
                         sublattice=sub_name)
    ldos.plot(label=sub_name)
pb.pltutils.legend()
plt.show()'''

plt.figure()
kpm_strain = pb.kpm(strained_model)
for sub_name in ['A1', 'B1', 'A2', 'B2']:
    ldos = kpm_strain.calc_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03,
                                position=[0, 0], sublattice=sub_name)
    ldos.plot(label=sub_name)
pb.pltutils.legend()
plt.show()

'''plt.figure()
strain_solv = pb.solver.arpack(strained_model, 30)
for sub_name in ['A1', 'B1', 'A2', 'B2']:
    ldos = strain_solv.calc_ldos(energies=np.linspace(-1, 1, 1500), broadening=0.03,
                                 position=[0, 0], sublattice=sub_name)
    ldos.plot(label=sub_name)
pb.pltutils.legend()
plt.show()'''

spatial_ldos = kpm_strain.calc_spatial_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03,  # eV
                                            shape=pb.regular_polygon(num_sides=6, radius=size, angle=pi))

smap = spatial_ldos.structure_map(0)
indices = smap.sublattices
x = smap.x
y = smap.y
data = smap.data
cond = [True if (i == 0) else False for i in indices]
x = x[cond]
y = y[cond]
data = data[cond]
# fig, (ax1, ax2) = plt.subplots(1)
plt.figure()
plt.scatter(x, y, c=data, s=0.5)
plt.colorbar()
plt.show()

