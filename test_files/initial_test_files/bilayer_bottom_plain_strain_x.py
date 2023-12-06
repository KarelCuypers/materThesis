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
        if z.all() == 0:
            ux = c * np.cos(k[0]*x + k[1]*y)
            uy = 0
            uz = 0
        else:
            ux = c * np.cos(k[0]*x + k[1]*y)
            uy = 0
            uz = 0
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


a1, a2 = graphene.bilayer().vectors[0], graphene.bilayer().vectors[1]

times_l1 = 5
times_l2 = 3
a = graphene.a_cc * sqrt(3)

l1_size = times_l1 * a

l2_size = times_l2 * sqrt(3) * a

period = 1
k = period * 2 * pi / l1_size * a1 / a

strained_model = pb.Model(
    graphene.bilayer(),
    unit_cell(l1= 2 * l1_size * a1/a, l2= 2 * l2_size * a2/a),
    pb.translational_symmetry(a1=l1_size, a2=l2_size),  # always needs some overlap with the rectangle
    sinusoidal_strain(0.04, k)
)

position = strained_model.system.xyz

strained_model.plot()
strained_model.lattice.plot_vectors(position=[0, 0])  # nm
plt.show()

solver = pb.solver.lapack(strained_model)

bands = solver.calc_bands(pi/a, -pi/a)
plt.figure()
for e in range(0, bands.num_bands):
    plt.scatter(bands.k_path, bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-3,3])
    # independently
plt.show()

lat = [[i[0]*times_l1, i[1]*(2*times_l1)] for i in strained_model.lattice.vectors]
full_lattice = pb.Lattice(a1=lat[0], a2=lat[1])

k_points = [i for i in full_lattice.brillouin_zone()]

kx = k_points[0][0]*3.5
ky = 5

kx_space = np.linspace(kx, -kx, 100)
ky_space = np.linspace(ky, -ky, 100)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
full_lattice.plot_brillouin_zone(color = 'g')

export_xyz("in_plain_xyz", position, l1_size * a1 / np.linalg.norm(a1), l2_size * a2 / np.linalg.norm(a2), np.array([0, 0, 1]), ['c'] * position.shape[0])
