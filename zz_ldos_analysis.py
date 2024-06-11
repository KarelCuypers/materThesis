import matplotlib.pyplot as plt
import numpy as np
import pybinding as pb
from functions.save_LDOS_xyz import save_LDOS_xyz
from pybinding.repository.graphene import a, a_cc, t


hbar = 4.136 * 10 ** (-15)  # eV*s
t_0 = 2.8  # eV
v_F = (3 * t_0 * a_cc) / (2*hbar)  # nm/s
t_perp = 0.035

n = 190
m = 200

x_times = 10
y_times = 600

ldos_path = 'C:/Users/Karel/Desktop/Master_Thesis/hetero_ldos_maps/'
sp_ldos_name = f'4_atom_supercell_size_{n}_over_{m}_x{x_times}_y{y_times}_hetro_strain_zz_AB'
sp_ldos = pb.load(f'{ldos_path}sp_ldos_{sp_ldos_name}.pbz')


idx = []
for i in range(len(sp_ldos.energy)):
    if 0.5 > sp_ldos.energy[i] > -0.5:
        idx.append(i)

idx_bottom = []
idx_top = []
top_atoms = []
bottom_atoms = []
for i in range(len(sp_ldos.positions[:, 1])):
    if sp_ldos.positions[i, 2] < 0:
        idx_bottom.append(i)
        bottom_atoms.append(sp_ldos.positions[i])
    else:
        idx_top.append(i)
        top_atoms.append(sp_ldos.positions[i])

top_atoms = np.array(top_atoms)
top_atoms = sorted(top_atoms, key=lambda x: x[1])

bottom_atoms = np.array(bottom_atoms)

idx_A2 = []
idx_B2 = []
for i in range(len(sp_ldos.positions[:, 1])):
    if 0.1 > sp_ldos.positions[i, 1] > 0 and sp_ldos.positions[i, 2] == 0:
        idx_B2.append(i)
    elif 0.1 < sp_ldos.positions[i, 1] and sp_ldos.positions[i, 2] == 0:
        idx_A2.append(i)
    elif -0.1 < sp_ldos.positions[i, 1] < 0 and sp_ldos.positions[i, 2] == 0:
        idx_A2.append(i)
    elif -0.1 > sp_ldos.positions[i, 1] and sp_ldos.positions[i, 2] == 0:
        idx_B2.append(i)

idx_A1 = []
idx_B1 = []
for i in range(len(sp_ldos.positions[:, 1])):
    if 0.1 > sp_ldos.positions[i, 1] > 0 and sp_ldos.positions[i, 2] != 0:
        idx_A1.append(i)
    elif sp_ldos.positions[i, 1] == 0 and sp_ldos.positions[i, 2] != 0:
        idx_B1.append(i)
    elif -0.1 > sp_ldos.positions[i, 1] and sp_ldos.positions[i, 2] != 0:
        idx_A1.append(i)
    elif 0.1 < sp_ldos.positions[i, 1] and sp_ldos.positions[i, 2] != 0:
        idx_B1.append(i)

filepath = f'{ldos_path}sp_ldos_{sp_ldos_name}'
Evals = sp_ldos.energy
coord = sp_ldos.positions
types = ['A'] * len(coord)
ldos = sp_ldos.data
#save_LDOS_xyz(filepath, Evals, coord, types, ldos, modulus_data=10)

ldos_bottom = sp_ldos.data[:, idx_bottom]
ldos_top = sp_ldos.data[:, idx_top]
A2_ldos = sp_ldos.data[:, idx_A2]
B2_ldos = sp_ldos.data[:, idx_B2]
A1_ldos = sp_ldos.data[:, idx_A1]
B1_ldos = sp_ldos.data[:, idx_B1]

dos_bottom = np.sum(ldos_bottom, axis=1)
dos_top = np.sum(ldos_top, axis=1)
dos_A2 = np.sum(A2_ldos, axis=1)
dos_B2 = np.sum(B2_ldos, axis=1)
dos_A1 = np.sum(A1_ldos, axis=1)
dos_B1 = np.sum(B1_ldos, axis=1)

'''plt.figure()
plt.plot(Evals, dos_top, label='top')
plt.plot(Evals, dos_bottom, label='bottom')
plt.legend()
plt.xlim(-0.6, 0.6)
plt.ylim(0, 15)
plt.show()

plt.figure()
plt.plot(Evals, dos_bottom, label='bottom')
plt.plot(Evals, dos_A1, label='A1')
plt.plot(Evals, dos_B1, label='B1')
plt.legend()
plt.xlim(-0.6, 0.6)
plt.ylim(0, 15)
plt.show()

plt.figure()
plt.plot(Evals, dos_top, label='top')
plt.plot(Evals, dos_A2, label='A2')
plt.plot(Evals, dos_B2, label='B2')
plt.legend()
plt.xlim(-0.6, 0.6)
plt.ylim(0, 15)
plt.show()

plt.figure()
plt.plot(Evals, dos_A1, label='A1')
plt.plot(Evals, dos_B1, label='B1')
plt.plot(Evals, dos_A2, label='A2')
plt.plot(Evals, dos_B2, label='B2')
plt.legend()
plt.xlim(-0.6, 0.6)
plt.ylim(0, 10)
plt.show()'''

c_x = (m - n) / n
E = np.pi / (3 * a) * c_x * v_F * hbar
E_dip = (hbar * v_F * 2*np.pi)**2 / (t_perp * (n*a*(1+c_x))**2)   #quadratic super lattice energy dip

plt.figure()
plt.plot(Evals, dos_bottom+dos_top, label='Total')
plt.legend()
plt.xlim(-1, 1)
plt.ylim(0, 50)
plt.axvline(E, color='r')
plt.axvline(E_dip, color='g')
plt.show()
