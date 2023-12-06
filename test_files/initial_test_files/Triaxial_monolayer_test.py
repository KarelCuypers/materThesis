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


def zigzag_hexagon(size):
    x0 = size / 2
    y0 = size * sqrt(3) / 2
    return pb.Polygon([[x0, y0], [2*x0, 0], [x0, -y0], [-x0, -y0], [-2*x0, 0], [-x0, y0]])


def armchair_hexagon(size):
    x0 = size * sqrt(3) / 2
    y0 = size / 2
    return pb.Polygon([[x0, y0], [x0, -y0], [0, -2*y0], [-x0, -y0], [-x0, y0], [0, 2*y0]])


B_ps = 1000

a_cc = graphene.a_cc * 10**(-9)
h = 4*10**(-15)
beta = 3.37
c = a_cc*B_ps/(4*h*beta) * 10**-9
print(c)

IB = sqrt(h/B_ps)*10**9
print(IB)

size = 15
zigzag = zigzag_hexagon(size)
armchair = armchair_hexagon(size)

zigzag_strained_model = pb.Model(
    graphene.monolayer_alt(),
    zigzag,
    triaxial_strain(c=c)
)

armchair_strained_model = pb.Model(
    graphene.monolayer(),
    armchair,
    triaxial_strain(c=c)
)

fig1, ax1 = plt.subplots()
zigzag_strained_model.plot()
plt.xlim([-12, 12])
plt.ylim([-12, 12])
plt.show()

fig2, ax2 = plt.subplots()
armchair_strained_model.plot()
plt.xlim([-12, 12])
plt.ylim([-12, 12])
plt.show()


plt.figure()
zigzag_kpm_strain = pb.kpm(zigzag_strained_model)
for sub_name in ['A', 'B']:
    ldos = zigzag_kpm_strain.calc_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03,
                                       position=[0, 0], sublattice=sub_name
                                       )
    ldos.plot(label=sub_name)
pb.pltutils.legend()
plt.show()

plt.figure()
armchair_kpm_strain = pb.kpm(armchair_strained_model)
for sub_name in ['A', 'B']:
    ldos = armchair_kpm_strain.calc_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03,
                                         position=[0, 0], sublattice=sub_name
                                         )
    ldos.plot(label=sub_name)
pb.pltutils.legend()
plt.show()

zigzag_spatial_ldos = zigzag_kpm_strain.calc_spatial_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03,
                                                          shape=zigzag
                                                          )

armchair_spatial_ldos = armchair_kpm_strain.calc_spatial_ldos(energy=np.linspace(-1, 1, 1500),
                                                              broadening=0.03, shape=armchair
                                                              )

E = 0.6
lv = 1

smap = zigzag_spatial_ldos.structure_map(E)
indices = smap.sublattices
x = smap.x
y = smap.y
data = smap.data
cond = [True if (i == lv) else False for i in indices]
x = x[cond]
y = y[cond]
data = data[cond]
plt.figure()
plt.scatter(x, y, c=data, s=0.5)
plt.colorbar()
plt.show()

smap = armchair_spatial_ldos.structure_map(E)
indices = smap.sublattices
x = smap.x
y = smap.y
data = smap.data
cond = [True if (i == lv) else False for i in indices]
x = x[cond]
y = y[cond]
data = data[cond]
plt.figure()
plt.scatter(x, y, c=data, s=0.5)
plt.colorbar()
plt.show()
