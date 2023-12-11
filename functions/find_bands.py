import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt


load_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/lattice_files/'
save_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/band_files/'
name = 'uniform_strain_xyz_test.xyz'
lattice = pb.load(f'{load_path}lattice_{name}')
print('Done')

model = pb.Model(lattice, pb.translational_symmetry())
solver = pb.solver.lapack(model)

# dispersion/band structure 2D/3D
bands = solver.calc_bands(3, -3, step=0.04)
#bands = solver.calc_wavefunction(3, -3, step=0.04).bands_disentangled

pb.save(bands, f'{save_path}bands_{name}')
print('Done')


# possible code for plotting a figure:

calculated_bands = pb.load(f'{save_path}bands_{name}')

fig, ax = plt.subplots()

for e in range(0, calculated_bands.num_bands):
    plt.scatter(calculated_bands.k_path, calculated_bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-3, 3])
    # independently
plt.show()

#[ax.plot(bands.k_path.as_1d(), en) for en in bands.energy.T]
#kx = 5
#ky = 5

#kx_space = np.linspace(kx, -kx, 100)
#ky_space = np.linspace(ky, -ky, 100)

#plt.figure()
#draw_contour(solver, kx_space, ky_space, round(len(bands.energy[0, :])/2), True)
