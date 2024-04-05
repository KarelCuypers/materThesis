import numpy as np


def calculate_surfaces(solver, kx_space, ky_space, band_index):
    def calculate_eigenvalues(kx, ky):
        solver.set_wave_vector([kx, ky])
        return solver.eigenvalues

    KX, KY = np.meshgrid(kx_space, ky_space)
    kx_list, ky_list = list(KX.flatten()), list(KY.flatten())

    conduction_E = np.zeros(shape=len(kx_space) * len(ky_space))
    valence_E = np.zeros(shape=len(kx_space) * len(ky_space))

    conduction_index = band_index
    valence_index = band_index-1

    for i, (kx, ky) in enumerate(zip(kx_list, ky_list)):

        eigenvalues = calculate_eigenvalues(kx, ky)

        valence_E[i] = eigenvalues[valence_index]
        conduction_E[i] = eigenvalues[conduction_index]

    conduction_E = conduction_E.reshape(len(ky_space), len(kx_space))
    valence_E = valence_E.reshape(len(ky_space), len(kx_space))

    diff = conduction_E - valence_E
    gap_size = min(map(min, diff))

    print(f'minimal size of the gap is {gap_size}')

    return KX, KY, conduction_E, valence_E, gap_size
