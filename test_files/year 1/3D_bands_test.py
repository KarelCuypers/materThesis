import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt


model = pb.Model(graphene.monolayer(),
                 pb.translational_symmetry()
                 )
#model.lattice.plot()
#plt.show()
#model.lattice.plot_brillouin_zone()
#plt.show()

solver = pb.solver.lapack(model)

num_list = np.linspace(-4, 4, 80)
X, Y = np.meshgrid(num_list, num_list)

band_1_array = []
for i in range(1, len(num_list)+1):
    a = [-4, i]
    b = [4, i]
    bands = solver.calc_bands(a, b)
    band_1 = bands.energy[:, :1]
    band_2 = bands.energy[:, 1:]
    print(band_1)
    flat_list = [item for sublist in band_1 for item in sublist]
    band_1_array.append(flat_list)

print(band_1_array)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contourf3D(X, Y, band_1_array)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

'''model.lattice.plot_brillouin_zone()
bands.plot_kpath(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()
bands.plot(point_labels=[r'$\Gamma$', 'R', 'S'])
plt.show()'''

