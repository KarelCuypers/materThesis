import pybinding as pb
import numpy as np
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


def hopping_modifier():
    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_0 = 0.184*a
        v_pi = t * np.exp(-(r - a/sqrt(3)) / r_0)
        v_sig = - 0.48 * np.exp(-(r - 0.335) / r_0)
        dz = z1 - z2

        return v_pi * (1 - (dz/r)**2) + v_sig * (dz/r)**2

    return strained_hopping


# apply strain
# strain_x = [[0.005, 0], [0.01, 0], [0.015, 0], [0.02, 0], [0.025, 0], [0.03, 0]]
# strain_x = [[0.03, 0], [0.035, 0], [0.04, 0], [0.045, 0], [0.05, 0]]
strain_x = [[0.03, 0]]
# strain_y = [[0, 0.005], [0, 0.01], [0, 0.015], [0, 0.02], [0, 0.025], [0, 0.03]]
# strain_y = [[0, 0.03], [0, 0.035], [0, 0.04], [0, 0.045], [0, 0.05]]
# strain_y = [[0, 0.025], [0, 0.03]]  # change frame to 0.3
# c = 0

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

for c in strain_x:
    if c[0] == 0:
        m = 1
        mode = 'armchair'
    else:
        m = 0
        mode = 'zigzag'
        print('yes')

    strained_model = pb.Model(
        strained_bilayer_lattice(gamma3=True, gamma4=True, strain=c, angle=angle),  # effect with gamma3 or 4 significantly different
        hopping_modifier(),
        pb.translational_symmetry()
    )

    a1, a2 = strained_model.lattice.vectors[0], strained_model.lattice.vectors[1]
    position = strained_model.system.xyz
    export_xyz("uniform_strain_xyz", position, a1, a2, np.array([0, 0, 1]), ['c'] * position.shape[0])

    plt.figure()
    strained_model.plot()
    strained_model.lattice.plot_vectors(position=[0, 0])  # nm
    plt.show()

    solver = pb.solver.lapack(strained_model)

    bands = solver.calc_wavefunction(K1[0] - 5, K1[0] + 5, step=0.001).bands_disentangled

    k_points = [bands.k_path[i][0] for i in range(0, len(bands.k_path))]
    fig, ax = plt.subplots()
    for e in range(0, bands.num_bands):
        plt.scatter(k_points, bands.energy[:, e], s=1,
                    color='g')  # methode to make much nicer looking plot or plot bands
        plt.ylim([-1, 1])
        # independently
    plt.scatter(K1[0], K1[1], s=1, color='r')
    plt.show()

    K1_strained = np.matmul([-4 * pi / (3 * sqrt(3) * a_cc) * (1 - c[0] / 2 - c[1] / 2), 0], rot)

    # K1 point 3D
    kx_max = K1_strained[0] + 0.2  # +0.2
    ky_max = K1_strained[1] + 0.2
    kx_min = K1_strained[0] - 0.2
    ky_min = K1_strained[1] - 0.2

    kx_space = np.linspace(kx_max, kx_min, 250)
    ky_space = np.linspace(ky_max, ky_min, 250)

    KX, KY, conduction_E, valence_E, gap_size = calculate_surfaces(solver, kx_space, ky_space, 2)
    gap.append(gap_size)

    draw_contour(KX, KY, conduction_E, valence_E, True, diff=False)
    plt.scatter(K1[0], K1[1], s=5, color='black')
    plt.scatter(K1_strained[0], K1_strained[1], s=5, color='b')
    plt.title(f'{c[m]*100}% {mode} strain conduction')
    #plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/{mode}_{c[m]}_band.png')
    plt.show()

    draw_contour(KX, KY, conduction_E, valence_E, True, diff=True)
    plt.scatter(K1[0], K1[1], s=5, color='black')
    plt.scatter(K1_strained[0], K1_strained[1], s=5, color='b')
    plt.title(f'{c[m]*100}% {mode} strain band diff')
    #plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/{mode}_{c[m]}_diff.png')
    plt.show()

    # dos fig around the K1 point

    kx_max = K1_strained[0] + 2  # + 0.5%
    ky_max = K1_strained[1] + 2
    kx_min = K1_strained[0] - 2
    ky_min = K1_strained[1] - 2

    kx_space = np.linspace(kx_max, kx_min, 250)
    ky_space = np.linspace(ky_max, ky_min, 250)

    # figure is correct just looks wierd because of the frame, should probably cut part away of x axis in final version

    dos, dos_energy, test = contour_dos(solver, kx_space, ky_space, 2, 150)
    plt.figure()
    plt.plot(dos_energy, dos, color='b')
    plt.title(f'{c[m] * 100}% {mode} strain dos')
    plt.xlabel('Energy')
    plt.ylabel('Number of states')
    #plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/{mode}_{c[m]}_dos.png')
    plt.show()

    '''kx_max = K1[0]
    ky_max = K2[1]
    kx_min = -K1[0]
    ky_min = -K2[1]

    kx_space = np.linspace(kx_max, kx_min, 250)
    ky_space = np.linspace(ky_max, ky_min, 250)

    dos, dos_energy = contour_dos(solver, kx_space, ky_space, 2, 100)
    plt.figure()
    plt.plot(dos_energy, dos, color='b')
    plt.title(f'{c[m] * 100}% {mode} strain dos')
    plt.xlabel('Energy')
    plt.ylabel('Number of states')
    plt.show()'''

plt.figure()
plt.scatter([c[m] for c in strain_x], gap)
plt.title(f'gap caused by {mode} strain')
plt.xlabel('strain')
plt.ylabel('gap')
#plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/{mode}_strain_gap.png')
