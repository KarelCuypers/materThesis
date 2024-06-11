import pybinding as pb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt
from functions.calculate_surfaces import calculate_surfaces
from functions.draw_contour import draw_contour
from functions.contour_dos import contour_dos
from functions.export_xyz import export_xyz


def strained_bilayer_lattice(gamma3=False, gamma4=False, onsite=(0, 0, 0, 0), strain=[0, 0], angle=0):

    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rot_3D = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    # change lattice vectors
    lat = pb.Lattice(
        a1=np.matmul([a/2 * (1+strain[0]), a/2 * sqrt(3) * (1+strain[1])], rot),
        a2=np.matmul([-a/2 * (1+strain[0]), a/2 * sqrt(3) * (1+strain[1])], rot)
    )

    c0 = 0.335  # [nm] interlayer spacing

    # change atom positions
    lat.add_sublattices(
        ('A1', np.matmul([0,  -a_cc/2 * (1+strain[1]),   0], rot_3D), onsite[0]),
        ('B1', np.matmul([0,   a_cc/2 * (1+strain[1]),   0], rot_3D), onsite[1]),
        ('A2', np.matmul([0,   a_cc/2 * (1+strain[1]), -c0], rot_3D), onsite[2]),
        ('B2', np.matmul([0, 3*a_cc/2 * (1+strain[1]), -c0], rot_3D), onsite[3])
    )

    t = graphene.t
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

ang = 0
angle = np.deg2rad(ang)  # y-direction is zigzag on rotate lattice
#angle = 0  # y-direction is armchair

# define constants
a = graphene.a_cc * sqrt(3)
a_cc = graphene.a_cc
t = graphene.t

gap = []

# dispersion/band structure 2D/3D
rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

Gamma = np.matmul([0, 0], rot)
K1 = np.matmul([-4 * pi / (3 * sqrt(3) * a_cc), 0], rot)  # K in paper
M = np.matmul([0, 2 * pi / (3 * a_cc)], rot)  # S in paper
K2 = np.matmul([2 * pi / (3 * sqrt(3) * a_cc), 2 * pi / (3 * a_cc)], rot)  # R in paper

no_skew_model = pb.Model(
    strained_bilayer_lattice(gamma3=False, gamma4=False, angle=angle),
    pb.translational_symmetry(),
    pb.force_phase(),
    pb.force_double_precision()
)

gamma_3_model = pb.Model(
    strained_bilayer_lattice(gamma3=True, gamma4=False, angle=angle),
    pb.translational_symmetry(),
    pb.force_phase(),
    pb.force_double_precision()
)

all_skew_model = pb.Model(
    strained_bilayer_lattice(gamma3=True, gamma4=True, angle=angle),  # effect with gamma3 or 4 significantly different
    pb.translational_symmetry(),
    pb.force_phase(),
    pb.force_double_precision()
)

a1, a2 = all_skew_model.lattice.vectors[0], all_skew_model.lattice.vectors[1]
position = all_skew_model.system.xyz
export_xyz("uniform_strain_xyz", position, a1, a2, np.array([0, 0, 1]), ['c'] * position.shape[0])

plt.figure()
all_skew_model.plot()
all_skew_model.lattice.plot_vectors(position=[0, 0])  # nm
plt.show()

all_skew_solver = pb.solver.lapack(all_skew_model)
gamma_3_solver = pb.solver.lapack(gamma_3_model)
no_skew_solver = pb.solver.lapack(no_skew_model)

matplotlib.rcParams.update({'font.size': 12})

#bands = solver.calc_wavefunction(K1[0] - 5, K1[0] + 5, step=0.1).bands_disentangled
all_skew_bands = all_skew_solver.calc_wavefunction(K1[0] - 0.05, K1[0] + 0.1, step=0.0001).bands_disentangled
gamma_3_bands = gamma_3_solver.calc_wavefunction(K1[0] - 0.05, K1[0] + 0.1, step=0.0001).bands_disentangled
no_skew_bands = no_skew_solver.calc_wavefunction(K1[0] - 0.05, K1[0] + 0.1, step=0.0001).bands_disentangled

k_points = [all_skew_bands.k_path[i][0] for i in range(0, len(all_skew_bands.k_path))]
plt.figure(figsize=(16/2, 10/2))
for e in range(0, all_skew_bands.num_bands):
    plt.plot(k_points, no_skew_bands.energy[:, e], color='b')
    plt.plot(k_points, gamma_3_bands.energy[:, e], color='y')
    plt.plot(k_points, all_skew_bands.energy[:, e], color='r')

plt.ylim([-0.002, 0.002])
plt.scatter(K1[0], K1[1], s=10, color='black')
plt.xlabel("k (1/nm)")
plt.ylabel('E (eV)')
plt.legend(['$\gamma_1, \gamma_2$', '$\gamma_3$','$\gamma_4$'])
plt.subplots_adjust(left=0.18)
plt.show()

# K1 point 3D
kx_max = K1[0] + 0.1  # +0.2
ky_max = K1[1] + 0.1
kx_min = K1[0] - 0.1
ky_min = K1[1] - 0.1

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

KX, KY, conduction_E, valence_E, gap_size = calculate_surfaces(all_skew_solver, kx_space, ky_space, 2)

plt.figure(figsize=(5, 5))
plt.scatter(K1[0], K1[1], s=5, color='black')
cmap = plt.get_cmap('coolwarm')
plt.contourf(KX, KY, conduction_E, 50, cmap=cmap)
plt.show()