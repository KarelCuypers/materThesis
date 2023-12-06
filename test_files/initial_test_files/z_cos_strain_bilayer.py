import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt
from export_xyz import export_xyz
from draw_contour import draw_contour


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
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w_intra = l / graphene.a_cc - 1
        w_inter = l / 0.335 - 1

        for i in np.subtract(z1, z2):  # changed because the wave in the z direction would break earlier code
            if i < 0.33:
                #print(energy * np.exp(-beta * w_intra))
                return energy * np.exp(-beta * w_intra)
            else:
                #print(0.48 * np.exp(-beta * w_inter))
                return 0.48 * np.exp(-beta * w_inter)

    return displacement, strained_hopping


def unit_cell(l1, l2):
    """Make the shape of a unit cell

    Parameters
    ----------
    l1, l2 : np.ndarray, np.ndarray
        Unit cell vectors.
    """

    A = np.array([0., 0.])
    B = np.array([l1[0], l1[1]])
    C = np.array([l1[0] + l2[0], l1[1] + l2[1]])
    D = np.array([l2[0], l2[1]])

    return pb.Polygon([A, B, C, D])


a1, a2 = graphene.monolayer().vectors[0], graphene.monolayer().vectors[1]

times_l1 = 3  # determines size of the unit cell of the graphene sheet
a = graphene.a_cc * sqrt(3)  # graphene unit cell length

l1_size = times_l1 * a  # lattice vector length * number of l1

l2_size = l1_size * 2  # has to be twice as big because 60° angle changes the phase

period = 1
k = period * 2 * pi / l1_size * a1 / a  # wavelength = 2*pi/lattice vector length * number of l1
# this makes sure there is exactly one period in the unit cell

strained_model = pb.Model(
    graphene.bilayer(),
    unit_cell(l1= 3.1 * l1_size * a1 / a, l2= 3.1 * l2_size * a2 / a),
    pb.translational_symmetry(a1=l1_size, a2=l2_size),  # always needs some overlap with the rectangle
    sinusoidal_strain(0.04, k)
)

position = strained_model.system.xyz

strained_model.plot()
strained_model.lattice.plot_vectors(position=[-0.6, 0.3])  # nm
plt.show()

solver = pb.solver.lapack(strained_model)

bands = solver.calc_bands(pi/a, -pi/a)
plt.figure()
for e in range(0, bands.num_bands):
    plt.scatter(bands.k_path, bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-3,3])
    # independently
plt.show()

# kpm_strain = pb.kpm(strained_model)
# for sub_name in ['A1', 'B1', 'A2', 'B2']:
#     ldos = kpm_strain.calc_ldos(energy=np.linspace(-1, 1, 1500), broadening=0.03,
#                                 position=[0, 0], sublattice=sub_name)
#     ldos.plot(label=sub_name)
# pb.pltutils.legend()
# plt.show()

lat = [[i[0]*(times_l1), i[1]*(2*times_l1)] for i in strained_model.lattice.vectors]
full_lattice = pb.Lattice(a1=lat[0], a2=lat[1])

k_points = [i for i in full_lattice.brillouin_zone()]

Gamma = [0, 0]
K1 = [k_points[0][0], k_points[0][1]]
M = [0, k_points[4][1]]
K2 = [k_points[5][0], k_points[5][1]]

bands = solver.calc_bands(-0.88*pi, 0.88*pi)
plt.figure()
bands.plot(point_labels=['K1', r'$\Gamma$'])

kx = k_points[0][0]*1.5
ky = 2
# plt.scatter(k_points[0][0])

kx_space = np.linspace(kx, -kx, 100)
ky_space = np.linspace(ky, -ky, 100)

draw_contour(solver, kx_space, ky_space, 2, True)
full_lattice.plot_brillouin_zone()

export_xyz("square_xyz", position, l1_size * a1 / np.linalg.norm(a1), l2_size * a2 / np.linalg.norm(a2), np.array([0, 0, 1]), ['c'] * position.shape[0])
