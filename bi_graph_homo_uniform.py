import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from functions.bilayer_4atom import bilayer_4atom
from math import pi, sqrt
from functions.export_xyz import export_xyz
from functions.four_atom_gating_term import four_atom_gating_term
from functions.draw_contour import draw_contour


def uniform_strain(c_x, c_y, angle):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = c_x * x
        uy = c_y * y
        uz = 0

        return (x + ux)*np.cos(angle), (y + uy)*np.sin(angle), z + uz

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
c_x = 0.08  #zz turn 60° = pi/3
c_y = 0.08  #ac turn 90° = pi/2

# define constants
a1, a2 = graphene.bilayer().vectors[0] * (1 + c_x), graphene.bilayer().vectors[1] * (1 + c_y)
a = graphene.a_cc * sqrt(3)
a_cc = graphene.a_cc
t = graphene.t

strained_model = pb.Model(
    graphene.bilayer(gamma3=True, gamma4=False, offset=[0.01, 0]), # effect with gamma3 or 4 significantly different
    pb.translational_symmetry(),
    uniform_strain(c_x, c_y, angle=pi/3)
)

position = strained_model.system.xyz

plt.figure()
strained_model.plot()
strained_model.lattice.plot_vectors(position=[0, 0])  # nm
plt.show()

solver = pb.solver.lapack(strained_model)

# dispersion/band structure 2D/3D

a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0] #K in paper
M = [0, 2*pi / (3*a_cc)] #S in paper
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)] #R in paper


plt.figure()
bands = solver.calc_bands(Gamma, K1, K2, step=0.001)
bands.plot(point_labels=[r'$\Gamma$', 'K1', 'K2'])
plt.show()

plt.figure()
bands = solver.calc_bands(Gamma, K2, M, step=0.01)
bands.plot(point_labels=[r'$\Gamma$', 'K2', 'M'])
plt.show()

bands = solver.calc_wavefunction(K1[0]-5, K1[0]+5, step=0.01).bands_disentangled
k_points = [bands.k_path[i][0] for i in range(0, len(bands.k_path))]
fig, ax = plt.subplots()
for e in range(0, bands.num_bands):
    plt.scatter(k_points, bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-1, 1])
    # independently
plt.scatter(K1[0], K1[1], s=1, color='r')
plt.show()

# K1 point 3D
kx_max = K1[0]+5
ky_max = K1[1]+5
kx_min = K1[0]-5
ky_min = K1[1]-5

kx_space = np.linspace(kx_max, kx_min, 100)
ky_space = np.linspace(ky_max, ky_min, 100)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)

# K2 point 3D
kx_max = K2[0]+5
ky_max = K2[1]+5
kx_min = K2[0]-5
ky_min = K2[1]-5

kx_space = np.linspace(kx_max, kx_min, 100)
ky_space = np.linspace(ky_max, ky_min, 100)

draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)

#export_xyz("uniform_strain_xyz", position, a1, a2, np.array([0, 0, 1]), ['c'] * position.shape[0])
