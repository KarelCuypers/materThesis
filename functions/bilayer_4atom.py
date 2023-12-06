import pybinding as pb
import matplotlib.pyplot as plt
from pybinding.repository.graphene.constants import a_cc, a, t, t_nn
from pybinding.repository import graphene


def bilayer_4atom(gamma3=False, gamma4=False, onsite=(0, 0, 0, 0)):

    lat = pb.Lattice(a1=[a, 0],
                     a2=[0, 3*a_cc])

    c0 = 0.335  # [nm] interlayer spacing

    lat.add_sublattices(
        # layer 1
        ('A1', [0,  -a_cc/2,   0], onsite[0]),
        ('B1', [0,   a_cc/2,   0], onsite[1]),
        # layer 2
        ('A2', [0,   a_cc/2, -c0], onsite[2]),
        ('B2', [0, 3*a_cc/2, -c0], onsite[3])

        # neighbours
        # Layer 1
        #('A3', [a / 2, -a_cc / 2 + 3 * a_cc / 2, 0]),
        #('B3', [a / 2, a_cc / 2 + 3 * a_cc / 2, 0]),
        # layer 2
        #('A4', [a / 2, a_cc / 2 + 3 * a_cc / 2, -c0]),
        #('B4', [a / 2, 3 * a_cc / 2 + 3 * a_cc / 2, -c0])
    )

    lat.add_aliases(
        # Layer 1
        ('A3', 'A1', [a / 2, -a_cc/2 + 3*a_cc/2, 0]),
        ('B3', 'B1', [a / 2, a_cc/2 + 3*a_cc/2 , 0]),
        # layer 2
        ('A4', 'A2', [a / 2, a_cc/2 + 3*a_cc/2, -c0]),
        ('B4', 'B2', [a / 2, 3*a_cc/2 + 3*a_cc/2 , -c0])
    )

    lat.register_hopping_energies({
        'gamma0': t,
        'gamma1': -0.48,
        'gamma3': -0.3,
        'gamma4': -0.04
    })

    lat.add_hoppings(
        # inside unit cell
        # layer 1
        ([ 0,  0], 'A1', 'B1', 'gamma0'),
        ([0, 0], 'B1', 'A3', 'gamma0'),
        ([0, 0], 'A3', 'B3', 'gamma0'),
        # layer 2
        ([0, 0], 'A2', 'B2', 'gamma0'),
        ([0, 0], 'B2', 'A4', 'gamma0'),
        ([0, 0], 'A4', 'B4', 'gamma0'),

        # between neighbouring unit cells
        #layer1
        ([-1, -1], 'A1', 'B3', 'gamma0'),
        ([0, -1], 'A1', 'B3', 'gamma0'),
        ([-1, 0], 'B1', 'A3', 'gamma0'),
        # layer 2
        ([-1, -1], 'A2', 'B4', 'gamma0'),
        ([0, -1], 'A2', 'B4', 'gamma0'),
        ([-1, 0], 'B2', 'A4', 'gamma0'),

        # interlayer
        ([0, 0], 'B1', 'A2', 'gamma1'),
        ([0, 0], 'B3', 'A4', 'gamma1')

    )

    # let op dit is nog voor gewone bilayer gaat dus nog aangepast moeten worden.

    if gamma3:
        lat.add_hoppings(
            ([1, 0], 'A1', 'A1', 'gamma3'),
            ([0, 0], 'A1', 'A3', 'gamma3'),
            ([-1, 0], 'A1', 'A3', 'gamma3'),
            ([-1, -1], 'A1', 'A3', 'gamma3'),
            ([0, -1], 'A1', 'B4', 'gamma3'),
            ([-1, -1], 'A1', 'B4', 'gamma3'),
            ([0, 1], 'A3', 'A1', 'gamma3'),
            ([0, 0], 'A1', 'A2', 'gamma3'),
            ([0, -1], 'A1', 'B2', 'gamma3'),

            ([0, 0], 'B1', 'B2', 'gamma3'),
            ([1, 0], 'B1', 'B1', 'gamma3'),
            ([0, 0], 'B1', 'B3', 'gamma3'),
            ([-1, 0], 'B1', 'B3', 'gamma3'),
            ([-1, -1], 'B1', 'B3', 'gamma3'),
            ([0, 1], 'B3', 'B1', 'gamma3'),


            ([1, 0], 'B3', 'B3', 'gamma3'),

            ([1, 0], 'A3', 'A3', 'gamma3'),

            ([1, 0], 'B4', 'B4', 'gamma3'),
            ([1, 0], 'A4', 'A4', 'gamma3'),


            ([0, 0], 'B4', 'B2', 'gamma3'),
            ([-1, 0], 'B2', 'B4', 'gamma3'),
            ([-1, -1], 'B2', 'B4', 'gamma3'),
            ([0, -1], 'B2', 'B4', 'gamma3'),

            ([1, 0], 'A2', 'A2', 'gamma3'),
            ([1, 0], 'B2', 'B2', 'gamma3'),

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


'''plt.figure()
lattice = graphene.monolayer_4atom()
lattice.plot()
plt.show()'''

'''plt.figure()
lattice = bilayer_4atom(gamma3=True)
lattice.plot()
plt.show()'''