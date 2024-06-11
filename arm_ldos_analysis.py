import matplotlib.pyplot as plt
import numpy as np
import pybinding as pb
from functions.save_LDOS_xyz import save_LDOS_xyz

n = 190
m = 200

x_times = 800
y_times = 10

ldos_path = 'C:/Users/Karel/Desktop/Master_Thesis/hetero_ldos_maps/'
sp_ldos_name = f'4_atom_supercell_size_{n}_over_{m}_x{x_times}_y{y_times}_hetro_strain_arm_AB'
sp_ldos = pb.load(f'{ldos_path}sp_ldos_{sp_ldos_name}.pbz')

idx = []
for i in range(len(sp_ldos.energy)):
    if 0.5 > sp_ldos.energy[i] > -0.5:
        idx.append(i)

filepath = f'{ldos_path}sp_ldos_{sp_ldos_name}'
Evals = sp_ldos.energy
coord = sp_ldos.positions
types = ['A'] * len(coord)
ldos = sp_ldos.data
#save_LDOS_xyz(filepath, Evals, coord, types, ldos, modulus_data=10)

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

ldos_top = sp_ldos.data[:, idx_top]
ldos_bottom = ldos[:, idx_bottom]

top_atoms = np.array(top_atoms)
idx_sorted_top_atoms = np.argsort(top_atoms[:, 1])
top_atoms = top_atoms[idx_sorted_top_atoms]
ldos_top_sorted = ldos_top[:, idx_sorted_top_atoms]

sort_ref = [0, 1, 2, 3]*len(ldos_top_sorted[1, :])

ldos_A2 = []
ldos_B2 = []
for i in range(len(ldos_top_sorted[1, :])):
    if sort_ref[i] == 0:
        ldos_B2.append(ldos_top_sorted[:, i].tolist())
    elif sort_ref[i] == 1:
        ldos_A2.append(ldos_top_sorted[:, i].tolist())
    elif sort_ref[i] == 2:
        ldos_B2.append(ldos_top_sorted[:, i].tolist())
    elif sort_ref[i] == 3:
        ldos_A2.append(ldos_top_sorted[:, i].tolist())
ldos_A2 = np.array(ldos_A2)
ldos_B2 = np.array(ldos_B2)

bottom_atoms = np.array(bottom_atoms)
idx_sorted_bottom_atoms = np.argsort(bottom_atoms[:, 1])
bottom_atoms = bottom_atoms[idx_sorted_bottom_atoms]
ldos_bottom_sorted = ldos_bottom[:, idx_sorted_bottom_atoms]

sort_ref = [0, 1, 2, 3]*len(ldos_top_sorted[1, :])

ldos_A1 = []
ldos_B1 = []
for i in range(len(ldos_bottom_sorted[1, :])):
    if sort_ref[i] == 0:
        ldos_B1.append(ldos_bottom_sorted[:, i].tolist())
    elif sort_ref[i] == 1:
        ldos_A1.append(ldos_bottom_sorted[:, i].tolist())
    elif sort_ref[i] == 2:
        ldos_B1.append(ldos_bottom_sorted[:, i].tolist())
    elif sort_ref[i] == 3:
        ldos_A1.append(ldos_bottom_sorted[:, i].tolist())
ldos_A1 = np.array(ldos_A1)
ldos_B1 = np.array(ldos_B1)

dos_bottom = np.sum(ldos_bottom, axis=1)
dos_bottom_sorted = np.sum(ldos_bottom_sorted, axis=1)
dos_top = np.sum(ldos_top, axis=1)
dos_top_sorted = np.sum(ldos_top_sorted, axis=1)
dos_A2 = np.sum(ldos_A2, axis=0)
dos_B2 = np.sum(ldos_B2, axis=0)
dos_A1 = np.sum(ldos_A1, axis=0)
dos_B1 = np.sum(ldos_B1, axis=0)

plt.figure()
plt.plot(Evals, dos_top, label='top')
plt.plot(Evals, dos_bottom, label='bottom')
plt.legend()
plt.xlim(-0.75, 0.75)
plt.ylim(0, 20)
plt.show()

plt.figure()
plt.plot(Evals, dos_bottom, label='bottom')
#plt.plot(Evals, dos_bottom_sorted, label='sorted bottom')
plt.plot(Evals, dos_A1, label='A1')
plt.plot(Evals, dos_B1, label='B1')
plt.legend()
plt.xlim(-0.6, 0.6)
plt.ylim(0, 15)
plt.show()

plt.figure()
plt.plot(Evals, dos_top, label='top')
#plt.plot(Evals, dos_top_sorted, label='sorted top')
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
plt.show()
