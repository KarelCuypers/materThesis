import numpy as np
import pybinding as pb
import matplotlib.pyplot as plt
from math import pi, sqrt

i = 50
c_x = 0.02

x = [round(0.0004*i, 4) for i in range(0, 51)]

load_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/ldos_size_16x50_sweep_sin/'

ldos = []
for c_x in x:
    name = f'supercell_size_{i}x16_{c_x}_strain_sin'
    calculated_ldos = pb.load(f'{load_path}ldos_{name}.pbz')
    y = calculated_ldos.variable
    ldos_data = calculated_ldos.data

    if c_x == 0.0:
        calculated_ldos.plot()

    ldos.append(ldos_data)

Z = np.array(ldos).T
#X, Y = np.meshgrid(x, y)

X = np.array(x)

fig, ax = plt.subplots()
plt.contourf(Z)
plt.show()

fig, ax = plt.subplots()
calculated_ldos.plot()
plt.show()