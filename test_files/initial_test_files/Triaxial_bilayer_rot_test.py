import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi, sqrt

pb.pltutils.use_style()


def bilayer_graphene(rot=0, gamma3=False, gamma4=False, onsite=(0, 0, 0, 0)):

    from math import sqrt
    from pybinding.repository.graphene.constants import a_cc, a, t
    c0 = 0.335  # [nm] interlayer spacing

    '''lat = pb.Lattice(
        a1=[a/2, a/2 * sqrt(3)],
        a2=[-a/2, a/2 * sqrt(3)]
    )'''

    lat = pb.Lattice(a1=[a/2*np.cos(rot) - a/2*sqrt(3)*np.sin(rot), a/2*np.sin(rot) + a/2*sqrt(3)*np.cos(rot)],
                     a2=[-a/2*np.cos(rot) - a/2*sqrt(3)*np.sin(rot), -a/2*np.sin(rot) + a/2*sqrt(3)*np.cos(rot)]
                     )

    lat.add_sublattices(
        ('A1', [0,  -a_cc/2,   0], onsite[0]),
        ('B1', [0,   a_cc/2,   0], onsite[1]),
        ('A2', [0,   a_cc/2, -c0], onsite[2]),
        ('B2', [0, 3*a_cc/2, -c0], onsite[3])
    )

    lat.register_hopping_energies({
        'gamma0': t,
        'gamma1': -0.4,
        'gamma3': -0.3,
        'gamma4': -0.04
    })

    lat.add_hoppings(
        # layer 1
        ([ 0,  0], 'A1', 'B1', 'gamma0'),
        ([ 0, -1], 'A1', 'B1', 'gamma0'),
        ([-1,  0], 'A1', 'B1', 'gamma0'),
        # layer 2
        ([ 0,  0], 'A2', 'B2', 'gamma0'),
        ([ 0, -1], 'A2', 'B2', 'gamma0'),
        ([-1,  0], 'A2', 'B2', 'gamma0'),
        # interlayer
        ([ 0,  0], 'B1', 'A2', 'gamma1')
    )

    if gamma3:
        lat.add_hoppings(
            ([0, 1], 'B2', 'A1', 'gamma3'),
            ([1, 0], 'B2', 'A1', 'gamma3'),
            ([1, 1], 'B2', 'A1', 'gamma3')
        )

    if gamma4:
        lat.add_hoppings(
            ([0, 0], 'A2', 'A1', 'gamma4'),
            ([0, 1], 'A2', 'A1', 'gamma4'),
            ([1, 0], 'A2', 'A1', 'gamma4')
        )

    lat.min_neighbors = 2
    return lat


model = pb.Model(
    graphene.monolayer_alt(),
    pb.translational_symmetry()
)

model2 = pb.Model(
    graphene.monolayer_4atom(),
    pb.translational_symmetry()
)

plt.figure()
model.plot()
model.lattice.plot_vectors(position=[-0, 0])  # nm
plt.show()

plt.figure()
model2.plot()
model2.lattice.plot_vectors(position=[-0, 0])  # nm
plt.show()