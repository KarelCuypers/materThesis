import numpy as np
import matplotlib.pyplot as plt


def draw_contour(solver, kx_space, ky_space, band_index, conduction=True, surf=False, dos_fig=False, n_bins=200):

    def calculate_surfaces(kx, ky):
        solver.set_wave_vector([kx, ky])
        return solver.eigenvalues

    KX, KY = np.meshgrid(kx_space, ky_space)
    kx_list, ky_list = list(KX.flatten()), list(KY.flatten())

    conduction_E = np.zeros(shape=len(kx_space) * len(ky_space))
    valence_E = np.zeros(shape=len(kx_space) * len(ky_space))

    conduction_index = band_index
    valence_index = band_index-1

    for i, (kx, ky) in enumerate(zip(kx_list, ky_list)):

        eigenvalues = calculate_surfaces(kx, ky)

        valence_E[i] = eigenvalues[valence_index]
        conduction_E[i] = eigenvalues[conduction_index]

    conduction_E = conduction_E.reshape(len(ky_space), len(kx_space))
    valence_E = valence_E.reshape(len(ky_space), len(kx_space))

    # 3D plot
    if surf:
        plt.figure(dpi=100)
        ax = plt.axes(projection='3d')
        cmap = plt.get_cmap('coolwarm')
        ax.plot_surface(KX, KY, conduction_E, cmap=cmap)
        ax.plot_surface(KX, KY, valence_E, cmap=cmap)
        plt.show()

    # contour plot
    plt.figure(dpi=100)
    cmap = plt.get_cmap('coolwarm')
    if conduction:
        plt.contourf(KX, KY, conduction_E, 50, cmap=cmap)
    else:
        plt.contourf(KX, KY, valence_E, 50, cmap=cmap)
    plt.colorbar()

    # dos figure attempt
    if dos_fig:
        c_bands = conduction_E.ravel().tolist()
        v_bands = valence_E.ravel().tolist()
        bands = c_bands + v_bands
        dos, e_edge = np.histogram(bands, bins=n_bins, density=True)
        dos_energy = [(e_edge[i] + e_edge[i+1])/2 for i in range(0, len(e_edge)-1)]
        plt.figure()
        plt.scatter(dos_energy, dos, s=1, color='b')
        plt.show()

    return conduction_E
