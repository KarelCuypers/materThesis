
import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from numpy import pi, sqrt

'''
def square_lattice(d, t):
    lat = pb.Lattice(a1=[d, 0], a2=[0, d])
    lat.add_sublattices(('A', [0, 0]))
    lat.add_hoppings(([0, 1], 'A', 'A', t),
                     ([1, 0], 'A', 'A', t))
    return lat


# we can quickly set a shorter unit length `d`
lattice = square_lattice(d=0.1, t=1)
lattice.plot()
plt.show()    # standard matplotlib show() function


def monolayer_graphene():
    a = 0.24595
    a_cc = 0.142
    t = -2.8

    lat = pb.Lattice(a1=[a, 0],
                     a2=[a/2, a/2*np.sqrt(3)])
    lat.add_sublattices(('A', [0, -a_cc/2]),
                        ('B', [0, a_cc/2]))
    lat.add_hoppings(
        # inside the main cell
        ([0, 0], 'A', 'B', t),
        # between neighboring cells
        ([1, -1], 'A', 'B', t),
        ([0, -1], 'A', 'B', t)
    )
    return lat


lattice = monolayer_graphene()
lattice.plot()
plt.show()
lattice.plot_brillouin_zone()
plt.show()
'''

'''
lattice = graphene.bilayer()
lattice.plot()
plt.show()

model = pb.Model(graphene.bilayer(),
                 pb.translational_symmetry()
                 )
model.plot()
plt.show()

print(model.hamiltonian)

solver = pb.solver.lapack(model)
print(solver.eigenvalues)

a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(K1, Gamma, M, K2)
bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
plt.show()
'''

pb.pltutils.use_style()


def ring(inner_radius, outer_radius):
    """A simple ring shape"""
    def contains(x, y, z):
        r = np.sqrt(x**2 + y**2)
        return np.logical_and(inner_radius < r, r < outer_radius)

    return pb.FreeformShape(contains, width=[2 * outer_radius, 2 * outer_radius])


model = pb.Model(
    graphene.monolayer_4atom(),
    ring(inner_radius=1.4, outer_radius=2),  # length in nanometers
    pb.translational_symmetry(a1=3.8, a2=False)
)


'''model = pb.Model(
    graphene.monolayer(),
    pb.rectangle(2, 2),
    pb.translational_symmetry(a1=1.2, a2=False)
)'''

model.plot()
plt.show()

solver = pb.solver.arpack(model, k=10)

a_cc = graphene.a_cc
Gamma = [0, 0]
K1 = [-4*pi / (3*sqrt(3)*a_cc), 0]
M = [0, 2*pi / (3*a_cc)]
K2 = [2*pi / (3*sqrt(3)*a_cc), 2*pi / (3*a_cc)]

bands = solver.calc_bands(-pi/3.8, pi/3.8)
# bands.plot(point_labels=['K', r'$\Gamma$', 'M', 'K'])
bands.plot()
plt.show()

kmp = pb.kpm(model)

ldos = kmp.calc_ldos(energy=np.linspace(-1, 1, 500), broadening=0.015, position=[0, 0])  # eV
ldos.plot()
plt.show()


