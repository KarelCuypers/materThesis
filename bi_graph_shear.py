import pybinding as pb
import numpy as np
import matplotlib.ticker
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt
from functions.calculate_surfaces import calculate_surfaces
from functions.draw_contour import draw_contour
from functions.contour_dos import contour_dos
from functions.four_atom_gating_term import four_atom_gating_term
from functions.export_xyz import export_xyz


def bilayer_shear(gamma3=False, gamma4=False, onsite=(0, 0, 0, 0), angle=0):
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rot_neg = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])

    lat = pb.Lattice(
        a1=np.matmul([a/2, a/2 * sqrt(3)], rot),
        a2=np.matmul([-a/2, a/2 * sqrt(3)], rot_neg)
    )

    c0 = 0.335  # [nm] interlayer spacing
    lat.add_sublattices(
        ('A1', [0,  -a_cc/2,   0], onsite[0]),
        ('B1', [0,   a_cc/2,   0], onsite[1]),
        ('A2', [0,   a_cc/2, -c0], onsite[2]),
        ('B2', [0, 3*a_cc/2, -c0], onsite[3])
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
    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_0 = 0.184*a
        v_pi = t * np.exp(-(r - a/sqrt(3)) / r_0)
        v_sig = - 0.48 * np.exp(-(r - 0.335) / r_0)
        dz = z1 - z2

        return v_pi * (1 - (dz/r)**2) + v_sig * (dz/r)**2

    return strained_hopping


# shear strain is defined as the difference in angle in rad
strain = [0.005, 0.01, 0.015, 0.02] #, -0.025, -0.03]

cm = 1 / 2.54

# strain = [-0.02]

mode = 'pos'
path = f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/shear_strain/{mode}_shear_strain'

# define constants
a = graphene.a_cc * sqrt(3)
a_cc = graphene.a_cc
t = graphene.t
a1, a2 = bilayer_shear().vectors[0], bilayer_shear().vectors[1]

temp = np.dot(a1, a2)/np.linalg.norm(a1)/np.linalg.norm(a2)
starting_angle = np.arccos(np.clip(temp, -1, 1))

Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]  # K in paper
M = [0, 2*pi / (3*a_cc)]  # S in paper
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]  # R in paper

gap = []

for c in strain:

    angle = c * starting_angle

    strained_model = pb.Model(
        bilayer_shear(gamma3=True, gamma4=True, angle=angle),  # effect with gamma3 or 4 significantly different
        pb.translational_symmetry(),
        uniform_strain(0, 0),
        #four_atom_gating_term(2*10**-3),
        pb.force_phase(),
        pb.force_double_precision()
    )

    #K1_strained_from_theory = [-4 * pi / (3 * sqrt(3) * a_cc), -2*angle]
    l1, l2 = strained_model.lattice.vectors
    points = strained_model.lattice.brillouin_zone()
    K1_strained = points[0]

    position = strained_model.system.xyz

    '''plt.figure()
    strained_model.plot()
    strained_model.lattice.plot_vectors(position=[0, 0])  # nm
    plt.show()'''

    solver = pb.solver.lapack(strained_model)

    # dispersion/band structure 2D

    # berry curvature calculation

    '''kx_max = K1_strained[0] + 0.2  # +0.2
    ky_max = K1_strained[1] + 0.2
    kx_min = K1_strained[0] - 0.2
    ky_min = K1_strained[1] - 0.2

    origin = [K1_strained[0] - 0.2, K1_strained[1] - 0.2]

    k_area = pb.make_area(*(0.4 * np.eye(2)), k_origin=origin, step=.05)
    wavefunction_array = solver.calc_wavefunction_area(k_area)
    berry_result = pb.berry.Berry(wavefunction_array, 2).calc_berry()

    the_berry_phase = berry_result.data_area[:-1, :-1, 0]/10000
    b_max = the_berry_phase.max()
    b_min = the_berry_phase.min()

    if abs(b_max) < abs(b_min):
        val = b_max
    else:
        val = b_min

    the_berry_phase_cut = np.flip(the_berry_phase)

    kx_space = np.linspace(kx_max, kx_min, len(the_berry_phase[0, :]))
    ky_space = np.linspace(ky_max, ky_min, len(the_berry_phase[:, 0]))

    KX, KY, conduction_E, valence_E, gap_size = calculate_surfaces(solver, kx_space, ky_space, 2)

    arr = conduction_E - valence_E
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] > 0.01:
                the_berry_phase_cut[i, j] = np.NaN

    plt.figure(figsize=(10 * cm, 7.5 * cm), dpi=600)
    mesh = plt.pcolormesh(berry_result.list_to_area(berry_result.k_path[:, 0]),
                          berry_result.list_to_area(berry_result.k_path[:, 1]),
                          the_berry_phase_cut, cmap='RdYlBu_r', rasterized=True
                          )

    clb = plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%.2f'))
    clb.ax.set_title('$x10^{-2}\mu$m$^2$')
    plt.xlim(kx_min, kx_max)
    plt.ylim(ky_min, ky_max)
    plt.title(f'{c * 100}%')
    plt.scatter(K1_strained[0], K1_strained[1], s=1, color='black')
    plt.annotate('$K_1\'$', [K1_strained[0], K1_strained[1]], c='black', xytext=(5, 5), textcoords='offset points')
    plt.savefig(f'{path}/{mode}_{c}_berry_with_1meV_gating.png')
    plt.show()'''

    # plot K1 point 3D

    '''draw_contour(KX, KY, conduction_E, valence_E, True, diff=True)
    plt.scatter(K1_strained[0], K1_strained[1], s=5, color='black')
    plt.annotate('$K_1\'$', [K1_strained[0], K1_strained[1]], c='black', xytext=(5, 5), textcoords='offset points')
    plt.title(f'{c * 100}%')
    plt.savefig(f'{path}/{mode}_{c}_diff_1meV_gating.png')
    plt.show()'''

    # dos fig around the K1 point

    kx_max = K1_strained[0] + 0.5  # + 0.5%
    ky_max = K1_strained[1] + 0.5
    kx_min = K1_strained[0] - 0.5
    ky_min = K1_strained[1] - 0.5

    kx_space = np.linspace(kx_max, kx_min, 500)
    ky_space = np.linspace(ky_max, ky_min, 500)

    # figure is correct just looks wierd because of the frame, should probably cut part away of x-axis in final version

    dos, dos_energy = contour_dos(solver, kx_space, ky_space, 2, 500)
    plt.figure(figsize=(10*cm, 7.5*cm), dpi=600)
    plt.plot(dos_energy, dos, color='black')
    plt.xlabel('Energy')
    plt.ylabel('Number of states')
    plt.xlim(-0.05, 0.05)
    plt.subplots_adjust(left=0.2, bottom=0.18)
    plt.title(f'{c * 100} %')
    plt.savefig(f'{path}/{mode}_{c}_dos.png')
    #plt.show()'''

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

'''plt.figure()
plt.scatter([c[m] for c in strain_x], gap)
plt.title(f'gap caused by {mode} strain')
plt.xlabel('strain')
plt.ylabel('gap')
#plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/{mode}_strain_gap.png')'''