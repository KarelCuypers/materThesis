import numpy as np
import matplotlib.pyplot as plt
import pybinding as pb
from math import sqrt, pi
from pybinding.repository import graphene


if __name__ == '__main__':

    model = pb.Model(graphene.monolayer(), pb.translational_symmetry())

    solver = pb.solver.lapack(model)

    def calculate_surfaces(kx, ky):
        solver.set_wave_vector([kx, ky])
        return solver.eigenvalues

    kx_space = np.linspace(-4 * pi / (3 * sqrt(3) * graphene.a_cc), 4 * pi / (3 * sqrt(3) * graphene.a_cc), 100)
    ky_space = np.linspace(-2*pi / (3*graphene.a_cc), 2*pi / (3*graphene.a_cc), 100)

    KX, KY = np.meshgrid(kx_space, ky_space)
    kx_list, ky_list = list(KX.flatten()), list(KY.flatten())

    conduction_E = np.zeros(shape=len(kx_space) * len(ky_space))
    valence_E = np.zeros(shape=len(kx_space) * len(ky_space))

    conduction_index = 1
    valence_index = 0

    for i, (kx, ky) in enumerate(zip(kx_list, ky_list)):

        eigenvalues = calculate_surfaces(kx, ky)

        valence_E[i] = eigenvalues[valence_index]
        conduction_E[i] = eigenvalues[conduction_index]

    conduction_E = conduction_E.reshape(len(ky_space), len(kx_space))
    valence_E = valence_E.reshape(len(ky_space), len(kx_space))

    plt.figure(dpi=200)

    cmap = plt.get_cmap('coolwarm')
    plt.contourf(valence_E, 50, cmap=cmap)
    plt.colorbar()
    plt.show()