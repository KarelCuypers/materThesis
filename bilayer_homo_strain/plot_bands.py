import pybinding as pb
import matplotlib.pyplot as plt
from math import pi, sqrt

load_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/lattice_files/'
save_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/band_files/'
name = 'bilayer4atom_no_strain'

calculated_bands = pb.load(f'{save_path}bands_{name}.pbz')

fig, ax = plt.subplots()
for e in range(0, calculated_bands.num_bands):
    plt.scatter(calculated_bands.k_path[:, 0], calculated_bands.energy[:, e], s=1, color = 'g') # methode to make much nicer looking plot or plot bands
    plt.ylim([-3, 3])
    # independently

#plt.scatter(4*pi/(3*sqrt(3)), 0, s=5, color = 'r')
plt.show()

lattice = pb.load(f'{load_path}lattice_{name}.pbz')

lattice.plot_brillouin_zone(color='r')
