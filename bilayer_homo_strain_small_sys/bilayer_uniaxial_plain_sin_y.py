import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from bilayer_4atom import bilayer_4atom
from math import pi, sqrt
from export_xyz import export_xyz
from unit_cell import unit_cell
from four_atom_gating_term import four_atom_gating_term
from draw_contour import draw_contour


def sinusoidal_strain(c, k, beta=3.37):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = c * np.cos(k[0]*x + k[1]*y)
        uz = 0
        return x + ux, y + uy, z + uz

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_0 = 0.184 * a
        v_pi = 2.7 * np.exp(-(r - a/sqrt(3)) / r_0)
        v_sig = - 0.48 * np.exp(-(r - 0.335) / r_0)
        dz = z1 - z2

        return v_pi * (1 - (dz/r)**2) + v_sig * (dz/r)**2

    return displacement, strained_hopping


a1, a2 = bilayer_4atom().vectors[0], bilayer_4atom().vectors[1]
a = graphene.a_cc * sqrt(3)

times_l1 = 5
times_l2 = 5

l1_size = times_l1 * np.linalg.norm(a1)

l2_size = times_l2 * np.linalg.norm(a2)

period = 1
k_y = period * 2 * pi / l2_size * a2 / np.linalg.norm(a2)

strained_model = pb.Model(
    bilayer_4atom(),
    unit_cell(l1=4 * times_l1 * a1, l2=4 * times_l1 * a2),
    pb.translational_symmetry(a1=2*l1_size, a2=2*l2_size),  # always needs some overlap with the rectangle
    sinusoidal_strain(0.08, k_y),
    four_atom_gating_term(0.5)
)

position = strained_model.system.xyz
lat = [times_l1*a1, times_l2*a2]
full_lattice = pb.Lattice(a1=lat[0], a2=lat[1])

strained_model.plot()
strained_model.lattice.plot_vectors(position=[0, 0])  # nm
plt.show()

solver = pb.solver.lapack(strained_model)

# density of states
dos = solver.calc_dos(energies=np.linspace(-2, 2, 200), broadening=0.05)

plt.figure()
dos.plot()

plt.figure()
for sub_name in ['A1', 'B1', 'A2', 'B2']:
    ldos = solver.calc_ldos(energies=np.linspace(-9, 9, 500), broadening=0.5,
                            position=[0, 0], sublattice=sub_name)
    ldos.plot(label=sub_name, ls="--" if sub_name == "B1" else "-")
pb.pltutils.legend()

# ldos map
ldos_map_1 = solver.calc_spatial_ldos(energy=0, broadening=0.5)
ldos_map_2 = solver.calc_spatial_ldos(energy=0.62, broadening=0.5)

xmin = -1
xmax = 7
ymin = -1
ymax = 11

plt.figure()
ax1 = plt.subplot(121)
ax1.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ldos_map_1.plot(ax=ax1)

ax2 = plt.subplot(122)
ax2.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ldos_map_2.plot(ax=ax2)

# dispersion

bands = solver.calc_bands(4, -4, step=0.05)
plt.figure()
for e in range(0, bands.num_bands):
    plt.scatter(bands.k_path, bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-3, 3])
    # independently
plt.show()

kx = 5
ky = 5
kx_space = np.linspace(kx, -kx, 100)
ky_space = np.linspace(ky, -ky, 100)

#plt.figure()
#draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
full_lattice.plot_brillouin_zone(color = 'r')

export_xyz("in_plain_sin_strain_xyz", position, l1_size * a1 / np.linalg.norm(a1),
           l2_size * a2 / np.linalg.norm(a2), np.array([0, 0, 1]), ['c'] * position.shape[0])
