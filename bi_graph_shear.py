import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt
from functions.calculate_surfaces import calculate_surfaces
from functions.draw_contour import draw_contour
from functions.contour_dos import contour_dos
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
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 0
        uy = 0
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

def dot(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return sqrt(dot(v, v))

def vec_angle(v1, v2):
  return np.arccos(dot(v1, v2) / (length(v1) * length(v2)))


angle = 0.0005

# define constants
a = graphene.a_cc * sqrt(3)
a_cc = graphene.a_cc
t = graphene.t
a1, a2 = bilayer_shear().vectors[0], bilayer_shear().vectors[1]


strained_model = pb.Model(
    bilayer_shear(gamma3=True, gamma4=True, angle=angle), # effect with gamma3 or 4 significantly different
    pb.translational_symmetry(),
    uniform_strain(0, 0)
)

position = strained_model.system.xyz

# according to wikipedia
a1_str, a2_str = strained_model.lattice.vectors[0], strained_model.lattice.vectors[1]
gamma_s = round((np.deg2rad(180) - vec_angle(a1_str, a2_str))/2, 3)

plt.figure()
strained_model.plot()
strained_model.lattice.plot_vectors(position=[0, 0])  # nm
plt.show()

solver = pb.solver.lapack(strained_model)

# dispersion/band structure 2D/3D
a_cc = graphene.a_cc

rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0] # K in paper
M = [0, 2*pi / (3*a_cc)] # S in paper
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)] # R in paper

bands = solver.calc_wavefunction(K1[0]-5, K1[0]+5, step=0.001).bands_disentangled

k_points = [bands.k_path[i][0] for i in range(0, len(bands.k_path))]
fig, ax = plt.subplots()
for e in range(0, bands.num_bands):
    plt.scatter(k_points, bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-1, 1])
    # independently
plt.scatter(K1[0], K1[1], s=1, color='r')
plt.show()

# K1 point 3D
kx_max = K1[0] + 0.2 # +0.2
ky_max = K1[1] + 0.2
kx_min = K1[0] - 0.2
ky_min = K1[1] - 0.2

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

KX, KY, conduction_E, valence_E = calculate_surfaces(solver, kx_space, ky_space, 2)

draw_contour(KX, KY, conduction_E, valence_E, True, diff=False)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.title(f'{gamma_s} shear strain conduction')
plt.xlabel('k_x')
plt.ylabel('k_y')
plt.show()

draw_contour(KX, KY, conduction_E, valence_E, True, diff=True)
plt.scatter(K1[0], K1[1], s=5, color='y')
plt.title(f'{gamma_s} shear strain band diff')
plt.xlabel('k_x')
plt.ylabel('k_y')
plt.show()

# dos fig around the K1 point

kx_max = K1[0] + 0.5 #+ 0.3
ky_max = K1[1] + 0.5
kx_min = K1[0] - 0.5
ky_min = K1[1] - 0.5

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

dos, dos_energy = contour_dos(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), 100)
plt.figure()
plt.plot(dos_energy, dos, color='b')
plt.title(f'{gamma_s} shear strain dos')
plt.xlabel('dos')
plt.ylabel('Energy')
plt.show()

kx_max = K1[0]
ky_max = K2[1]
kx_min = -K1[0]
ky_min = -K2[1]

kx_space = np.linspace(kx_max, kx_min, 250)
ky_space = np.linspace(ky_max, ky_min, 250)

dos, dos_energy = contour_dos(solver, kx_space, ky_space, 2, 100)
plt.figure()
plt.plot(dos_energy, dos, color='b')
plt.title(f'{angle} shear strain rot dos')
plt.xlabel('dos')
plt.ylabel('Energy')
plt.show()

export_xyz("shear_strain_xyz", position, a1, a2, np.array([0, 0, 1]), ['c'] * position.shape[0])
