import numpy as np
import pybinding as pb
from concurrent import futures
import matplotlib.pyplot as plt
from functions.draw_contour import draw_contour


n = 195
m = 200
x_times = 1
y_times = 1
xyz_name = f"hetro_{n}_over_{m}_coord_zz_AB"

def calculate_eigenvalues(kx, ky):
    complete_lattice = pb.load(f'lattice_{xyz_name}.pbz')

    model = pb.Model(complete_lattice,
                         pb.primitive(x_times, y_times),
                         pb.translational_symmetry(a1=x_times * complete_lattice.vectors[0][0],
                                                   a2=y_times * complete_lattice.vectors[1][1])
                         )

    solver = pb.solver.arpack(model, k=4, sigma=0)
    solver.set_wave_vector([kx, ky])
    return solver.eigenvalues


def calculate_surface_parallel(kx_space, ky_space, band_index):

    KX, KY = np.meshgrid(kx_space, ky_space)
    kx_list, ky_list = np.array(KX.flatten()), np.array(KY.flatten())

    conduction_E = np.zeros(shape=len(kx_space) * len(ky_space))
    valence_E = np.zeros(shape=len(kx_space) * len(ky_space))

    conduction_index = band_index
    valence_index = band_index-1

    idx = np.arange(0, len(kx_list))

    with futures.ProcessPoolExecutor() as executor:
        for i, result in zip(idx, executor.map(calculate_eigenvalues, kx_list, ky_list)):
            eigenvalues = result
            print(eigenvalues)

            valence_E[i] = eigenvalues[valence_index]
            conduction_E[i] = eigenvalues[conduction_index]

    conduction_E = conduction_E.reshape(len(ky_space), len(kx_space))
    valence_E = valence_E.reshape(len(ky_space), len(kx_space))

    diff = abs(conduction_E - valence_E)
    gap_size = min(map(min, diff))

    print(f'minimal size of the gap is {gap_size}')

    return KX, KY, conduction_E, valence_E, gap_size


if __name__ == '__main__':
    print('started caclc')

    '''complete_lattice = pb.load(f'lattice_{xyz_name}.pbz')

    points = complete_lattice.brillouin_zone()
    K1_strained = points[0]'''

    kx_max = 0 + 0.5  # +0.2
    ky_max = 0 + 0.5
    kx_min = 0 - 0.5
    ky_min = 0 - 0.5

    '''kx_max = 1.4 + 0.5  # +0.2
    ky_max = 0 + 0.5
    kx_min = 1.4 - 0.5
    ky_min = 0 - 0.5'''

    kx_space = np.linspace(kx_max, kx_min, 100)
    ky_space = np.linspace(ky_max, ky_min, 100)

    KX, KY, conduction_E, valence_E, gap_size = calculate_surface_parallel(kx_space, ky_space, 2)
    np.save(f'cond_{xyz_name}.npy', conduction_E)
    np.save(f'valence_{xyz_name}.npy', valence_E)

    '''draw_contour(KX, KY, conduction_E, valence_E, True, surf=True, diff=False)
    # plt.scatter(K1[0], K1[1], s=5, color='r')
    # plt.scatter(K1_strained[0], K1_strained[1], s=5, color='black')
    plt.annotate('$K_1\'$', [K1_strained[0], K1_strained[1]], c='black', xytext=(5, 5), textcoords='offset points')
    #plt.title(f'{c[tag] * 100}%')
    # plt.savefig(f'{path}/{mode}_{c[m]}_diff.png')
    plt.show()'''
