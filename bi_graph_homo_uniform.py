import os
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
from pybinding.repository import graphene
from math import pi, sqrt
from functions.create_lattice import create_lattice
from functions.draw_contour import draw_contour
from functions.export_xyz import export_xyz
from functions.calculate_surfaces import calculate_surfaces
from functions.four_atom_gating_term import four_atom_gating_term
from functions.contour_dos import contour_dos

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1


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


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 12})
    cm = 1 / 2.54

    # apply strain
    # strain_x = [[0.005, 0], [0.01, 0], [0.015, 0], [0.02, 0], [0.025, 0], [0.03, 0]]
    # strain_x = [[0.0025, 0], [0.005, 0], [0.0075, 0], [0.01, 0]]
    # strain_y = [[0, 0.025], [0, 0.03]]  # change frame to 0.3
    # strain_y = [[0, 0.005], [0, 0.01], [0, 0.015], [0, 0.02]]
    # strain_y = [[0, 0], [0, 0.0025], [0, 0.005], [0, 0.0075], [0, 0.01]]
    strain_y = [[0, 0]]

    ang = 0
    angle = np.deg2rad(ang)  # y-direction is zigzag on rotate lattice
    # angle = 0  # y-direction is armchair

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

    for c in strain_y:
        if c[0] == 0:
            tag = 1
            mode = 'armchair'
            tit = 'AC'
            path = f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/homo_uniform_strain/{mode}'
        else:
            tag = 0
            mode = 'zigzag'
            tit = 'ZZ'
            path = f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/homo_uniform_strain/{mode}'

        print(mode)

        strained_model = pb.Model(
            strained_bilayer_lattice(gamma3=True, gamma4=True, strain=c, angle=angle),
            # effect with gamma3 or 4 significantly different
            hopping_modifier(),
            pb.translational_symmetry(),
            #four_atom_gating_term(2*10**-3),
            pb.force_phase(),
            pb.force_double_precision()
        )

        a1, a2 = strained_model.lattice.vectors[0], strained_model.lattice.vectors[1]
        position = strained_model.system.xyz
        xyz_name = f"test_{mode}"
        export_xyz(xyz_name, position, a1, a2, np.array([0, 0, 1]), ['A'] * position.shape[0])

        complete_lattice = create_lattice(xyz_name)

        solver = pb.solver.lapack(strained_model)

        '''model = pb.Model(complete_lattice,
                         pb.translational_symmetry())
        solver = pb.solver.lapack(model)'''

        K1_strained = np.matmul([-4 * pi / (3 * sqrt(3) * a_cc) * (1 - c[0] / 2 - c[1] / 2), 0], rot)
        '''l1, l2 = strained_model.lattice.vectors
        points = strained_model.lattice.brillouin_zone()
        K1_strained = points[0]'''

        j = 0.2

        kx_max = K1_strained[0] + j  # +0.2
        ky_max = K1_strained[1] + j
        kx_min = K1_strained[0] - j
        ky_min = K1_strained[1] - j

        # berry curvature calculation

        '''origin = [kx_min, ky_min]

        k_area = pb.make_area(*(2*j * np.eye(2)), k_origin=origin, step=.05)
        wavefunction_array = solver.calc_wavefunction_area(k_area)
        berry_result = pb.berry.Berry(wavefunction_array, 2).calc_berry()

        the_berry_phase = berry_result.data_area[:-1, :-1, 0]/10000  #
        b_max = the_berry_phase.max()
        b_min = the_berry_phase.min()

        if abs(b_max) > abs(b_min):
            val = abs(b_max)
        else:
            val = abs(b_min)

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
                       np.flip(the_berry_phase_cut), cmap='RdYlBu_r', rasterized=True)
        clb = plt.colorbar(format=matplotlib.ticker.FormatStrFormatter('%.1f'))
        # clb.ax.set_yticklabels(['-', '0', '+'])
        clb.ax.set_title('$x10^{-2}\mu$m$^2$')
        plt.xlim(kx_min, kx_max)
        plt.ylim(ky_min, ky_max)
        plt.title(f'{c[tag]*100}%')
        plt.scatter(K1_strained[0], K1_strained[1], s=1, color='black')
        plt.annotate('$K_1\'$', [K1_strained[0], K1_strained[1]], c='black', xytext=(5, 5), textcoords='offset points')
        plt.savefig(f'{path}/new_{mode}_{c[tag]}_berry_with_gating_2meV.png')

        #plt.savefig(f'{path}/just_sign_{mode}_{c[tag]}_berry_with_gating_2meV.png')'''

        # plot K1 point 3D
        # remove gap !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # surface parallel test

        kx_space = np.linspace(kx_max, kx_min, 250)
        ky_space = np.linspace(ky_max, ky_min, 250)

        from functions.calculate_surfaces_parallel import calculate_surface_parallel

        #KX, KY, conduction_E, valence_E, gap_size = calculate_surface_parallel(kx_space, ky_space, 2)'''
        KX, KY, conduction_E, valence_E, gap_size = calculate_surfaces(solver, kx_space, ky_space, 2)

        draw_contour(KX, KY, conduction_E, valence_E, True, diff=True)
        #plt.scatter(K1_strained[0], K1_strained[1], s=5, color='black')
        #plt.annotate('$K_1\'$', [K1_strained[0], K1_strained[1]], c='black', xytext=(5, 5), textcoords='offset points')
        plt.title(f'{c[tag] * 100}%')
        #plt.title(f'Trigonal warping')
        plt.savefig(f'{path}/{mode}_{c[tag]}_diff.png')

        # dos fig around the K1 point

        '''kx_max = K1_strained[0] + 0.3  # + 0.5%
        ky_max = K1_strained[1] + 0.3
        kx_min = K1_strained[0] - 0.3
        ky_min = K1_strained[1] - 0.3

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
        plt.title(f'{c[tag] * 100}%')
        plt.savefig(f'{path}/{mode}_{c[tag]}_dos_improved.png')
        plt.show()'''

    '''plt.figure(figsize=(6, 4), dpi=600)
    plt.scatter([c[m]*100 for c in strain_y], np.array(gap)*1000)
    plt.xlabel('strain %')
    plt.ylabel('gap (meV)')
    plt.subplots_adjust(left=0.15, bottom=0.18)
    plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/band_structure_plots/{mode}_strain_gap_higher.png')'''
