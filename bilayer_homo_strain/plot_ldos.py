import numpy as np
import matplotlib.colors as colors
import pybinding as pb
import matplotlib.pyplot as plt
from math import pi, sqrt


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
        self.vcenter = vcenter
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        # Note also that we must extrapolate beyond vmin/vmax
        x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
        return np.ma.masked_array(np.interp(value, x, y,
                                            left=-np.inf, right=np.inf))

    def inverse(self, value):
        y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
        return np.interp(value, x, y, left=-np.inf, right=np.inf)


repeat = 50
c_x = 0.02

#x = [round(0.0004*i, 4) for i in range(0, 51)]
x = [0.01, 0.02]

load_path = ('C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/'
             'ldos_size_16x50_sweep_sin_y/')

ldos = []

for c_x in x:
    name = f'supercell_size_{repeat}x16_{c_x}_strain_sin_y'
    calculated_ldos = pb.load(f'{load_path}ldos_{name}.pbz')
    ldos_data = calculated_ldos.data
    print(c_x)

    if c_x == 0.01:
        calculated_ldos.plot()

    ldos.append(ldos_data)

'''lattice = pb.load(f'{load_path}lattice_{name}.pbz')
model = pb.Model(lattice)

hbar = 4.136*10**(-15) #eV*s
t_0 = 2.7 #eV
a = 1.42 * 0.1 #nm
v_F = 3/2 * t_0 * a/hbar
Gnorm = np.linalg.norm(model.lattice.reciprocal_vectors()[1])

#E = hbar * v_F * Gnorm/2
E = 3/2*t_0*a*Gnorm/2
print(E)'''

Z_data = np.array(ldos).T
Z = NormalizeData(Z_data)
y = calculated_ldos.variable

idx = [i for i, v in enumerate(y) if abs(v) < 0.2]
y = y[idx]

fig, ax = plt.subplots()
calculated_ldos.plot()
#plt.xlim(-0.5, 0.5)
#plt.ylim(0, 70000)
#plt.axvline(x=E, color='r', label='axvline-full height')
#plt.axvline(x=3*E, color='r', label='axvline-full height')
#plt.axvline(x=7/2*E, color='r', label='axvline-full height')
#plt.axvline(x=5*E, color='r', label='axvline-full height')
plt.show()

fig, ax = plt.subplots()

#midnorm = MidpointNormalize(vmin=0, vcenter=30000, vmax=50000)
#plt.pcolormesh(Z, norm=colors.LogNorm(vmin=Z.min(), vmax=70000), cmap='jet')
#plt.pcolormesh(Z, norm=midnorm, cmap='jet')

plt.pcolormesh(Z[idx], cmap='jet')
#plt.pcolormesh(Z, cmap='jet')
default_x_ticks = range(len(x))
default_y_ticks = range(len(y))
#new_y = np.linspace(-0.2, 0.2, 10000)
plt.xticks(default_x_ticks, x, rotation=90)
plt.yticks(default_y_ticks, [round(i, 2) for i in y])
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=10)
cbar = plt.colorbar()
plt.show()

