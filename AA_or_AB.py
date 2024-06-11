import pybinding as pb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pybinding.repository.graphene import a_cc

matplotlib.rcParams.update({'font.size': 12})
cm = 1 / 2.54

hbar = 4.136 * 10 ** (-15)  # eV*s
t_0 = 2.8  # eV
v_F = (3 * t_0 * a_cc) / (2*hbar)
g1 = 0.381 #eV

m = g1 / (2*v_F)

E = 4*m/(2*np.pi * hbar**2)

atoms_layer_1 = 200 * 8 * 10 * 600
atoms_layer_2 = 200 * 8 * 10 * 600

path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/hetero_dos_calculations/zigzag/10x600_AB/'
name = f'4_atom_supercell_size_200_over_200_x10_y600_hetro_strain_zz_AB'

dos_AB = pb.load(f'{path}dos_{name}.pbz')
show_dos_AB = dos_AB.data / (atoms_layer_1 + atoms_layer_2)
y_AB = dos_AB.variable

path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/hetero_dos_calculations/zigzag/10x600_AA/'
name = f'4_atom_supercell_size_200_over_200_x10_y600_hetro_strain_zz_AA'

dos_AA = pb.load(f'{path}dos_{name}.pbz')
show_dos_AA = dos_AA.data / (atoms_layer_1 + atoms_layer_2)
y_AA = dos_AA.variable

plt.figure(figsize=(10 * cm, 7.5 * cm), dpi=600)
plt.plot(dos_AB.variable, show_dos_AB, label='AB stacking')
plt.plot(dos_AA.variable, show_dos_AA, label='AA stacking')
#plt.legend()
#plt.xlim(-0.5, 0.5)
#plt.ylim(0, 0.08*10**-1)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xlabel('E (eV)')
plt.ylabel('DOS per atom')
plt.subplots_adjust(left=0.15, bottom=0.18)
plt.savefig(f'C:/Users/Karel/Desktop/Master_Thesis/AB_AA_stack_example_alt_full')