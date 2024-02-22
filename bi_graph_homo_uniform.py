import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt
from functions.draw_contour import draw_contour
from functions.contour_dos import contour_dos

def bilayer_rotated(gamma3=False, gamma4=False, onsite=(0, 0, 0, 0), angle=0):

    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rot_3D = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    lat = pb.Lattice(
        a1=np.matmul([a/2, a/2 * sqrt(3)], rot),
        a2=np.matmul([-a/2, a/2 * sqrt(3)], rot)
    )

    c0 = 0.335  # [nm] interlayer spacing
    lat.add_sublattices(
        ('A1', np.matmul([0,  -a_cc/2,   0], rot_3D), onsite[0]),
        ('B1', np.matmul([0,   a_cc/2,   0], rot_3D), onsite[1]),
        ('A2', np.matmul([0,   a_cc/2, -c0], rot_3D), onsite[2]),
        ('B2', np.matmul([0, 3*a_cc/2, -c0], rot_3D), onsite[3])
    )

    lat.register_hopping_energies({
        'gamma0': t,
        'gamma1': -0.4,
        'gamma3': -0.3,
        'gamma4': -0.04
    })

    lat.add_hoppings(
        # layer 1
        ([ 0,  0], 'A1', 'B1', 'gamma0'),
        ([ 0, -1], 'A1', 'B1', 'gamma0'),
        ([-1,  0], 'A1', 'B1', 'gamma0'),
        # layer 2
        ([ 0,  0], 'A2', 'B2', 'gamma0'),
        ([ 0, -1], 'A2', 'B2', 'gamma0'),
        ([-1,  0], 'A2', 'B2', 'gamma0'),
        # interlayer
        ([ 0,  0], 'B1', 'A2', 'gamma1')
    )

    if gamma3:
        lat.add_hoppings(
            ([0, 1], 'B2', 'A1', 'gamma3'),
            ([1, 0], 'B2', 'A1', 'gamma3'),
            ([1, 1], 'B2', 'A1', 'gamma3')
        )

    if gamma4:
        lat.add_hoppings(
            ([0, 0], 'A2', 'A1', 'gamma4'),
            ([0, 1], 'A2', 'A1', 'gamma4'),
            ([1, 0], 'A2', 'A1', 'gamma4')
        )

    lat.min_neighbors = 2
    return lat


def uniform_strain(c_x, c_y):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = c_x * x
        uy = c_y * y
        uz = 0

        return x + ux, y + uy, z + uz

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_0 = 0.184*a
        v_pi = 2.7 * np.exp(-(r - a/sqrt(3)) / r_0)
        v_sig = - 0.48 * np.exp(-(r - 0.335) / r_0)
        dz = z1 - z2

        return v_pi * (1 - (dz/r)**2) + v_sig * (dz/r)**2

    return displacement, strained_hopping


# apply strain
c_x = 0  # zz
c_y = 0.08  # ac

#angle = pi/3
angle = 0

# define constants
a1, a2 = graphene.bilayer().vectors[0] * (1 + c_x), graphene.bilayer().vectors[1] * (1 + c_y)
a = graphene.a_cc * sqrt(3)
a_cc = graphene.a_cc
t = graphene.t

strained_model = pb.Model(
    bilayer_rotated(gamma3=True, gamma4=True, angle=angle), # effect with gamma3 or 4 significantly different
    pb.translational_symmetry(),
    uniform_strain(c_x, c_y)
)

position = strained_model.system.xyz

plt.figure()
strained_model.plot()
strained_model.lattice.plot_vectors(position=[0, 0])  # nm
plt.show()

solver = pb.solver.lapack(strained_model)

# dispersion/band structure 2D/3D
a_cc = graphene.a_cc

rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

Gamma = np.matmul([0, 0], rot)
K1 = np.matmul([-4*pi / (3*sqrt(3)*a_cc), 0], rot) # K in paper
M = np.matmul([0, 2*pi / (3*a_cc)], rot) # S in paper
K2 = np.matmul([2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)], rot) # R in paper

'''plt.figure()
bands = solver.calc_bands(Gamma, K1, K2, step=0.001)
bands.plot(point_labels=[r'$\Gamma$', 'K1', 'K2'])
plt.show()

plt.figure()
bands = solver.calc_bands(Gamma, K2, M, step=0.001)
bands.plot(point_labels=[r'$\Gamma$', 'K2', 'M'])
plt.show()

plt.figure()
start = [-4*pi / (3*sqrt(3)*a_cc)+10, 0]
stop = [-4*pi / (3*sqrt(3)*a_cc)-10, 0]
bands = solver.calc_bands(start, K1, stop, step=0.001)
bands.plot(point_labels=['start', 'K1', 'stop'])
plt.show()'''

bands = solver.calc_wavefunction(K1[0]-5, K1[0]+5, step=0.001).bands_disentangled

k_points = [bands.k_path[i][0] for i in range(0, len(bands.k_path))]
fig, ax = plt.subplots()
for e in range(0, bands.num_bands):
    plt.scatter(k_points, bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-1, 1])
    # independently
plt.scatter(K1[0], K1[1], s=1, color='r')
plt.scatter(K1[0] * (1-c_y), K1[1], s=1, color='r')
plt.show()

# For ac-strain

'''# K1 point 3D
kx_max = K1[0] + 0.15
ky_max = K1[1] + 0.1
kx_min = K1[0] - 0.1
ky_min = K1[1] - 0.1

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.show()

# K1 close up
kx_max = K1[0] + 0.02
ky_max = K1[1] + 0.1
kx_min = K1[0] - 0.05
ky_min = K1[1] - 0.1

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.show()

# K1 wide 
kx_max = K1[0] + 0.3
ky_max = K1[1] + 0.3
kx_min = K1[0] - 0.3
ky_min = K1[1] - 0.3

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.show()'''

# for zz-strain

'''# K1 point 3D
kx_max = K1[0] + 0.1
ky_max = K1[1] + 0.1
kx_min = K1[0] - 0.15
ky_min = K1[1] - 0.1

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.show()

# K1 close up
kx_max = K1[0] + 0.02
ky_max = K1[1] + 0.1
kx_min = K1[0] - 0.08
ky_min = K1[1] - 0.1

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.show()

# K1 wide
kx_max = K1[0] + 0.3
ky_max = K1[1] + 0.3
kx_min = K1[0] - 0.3
ky_min = K1[1] - 0.3

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.show()'''

kx_max = K1[0] + 5
ky_max = K1[1] + 5
kx_min = K1[0] - 5
ky_min = K1[1] - 5

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.show()

# dos fig

kx_max = K1[0] + 0.05
ky_max = K1[1] + 0.05
kx_min = K1[0] - 0.05
ky_min = K1[1] - 0.05

kx_space = np.linspace(kx_max, kx_min, 500)
ky_space = np.linspace(ky_max, ky_min, 500)

test = draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True, dos_fig=True)

#export_xyz("uniform_strain_xyz", position, a1, a2, np.array([0, 0, 1]), ['c'] * position.shape[0])
