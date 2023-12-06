"""Build the simplest model of bilayer graphene and compute its band structure"""
import pybinding as pb
import matplotlib.pyplot as plt
from math import sqrt, pi

pb.pltutils.use_style()


def bilayer_graphene():
    """Bilayer lattice in the AB-stacked form (Bernal-stacked)

    This is the simplest model with just a single intralayer and a single interlayer hopping.
    """
    a = 0.24595   # [nm] unit cell length
    a_cc = 0.142  # [nm] carbon-carbon distance
    c0 = 0.335    # [nm] interlayer spacing

    lat = pb.Lattice(a1=[a/2, a/2 * sqrt(3)], a2=[a/2, -a/2 * sqrt(3)])

    lat.add_sublattices(
        ('A1', [0,  -a_cc/2,   0]),  # sublatices of layer 1
        ('B1', [0,   a_cc/2,   0]),
        ('A2', [0,   a_cc/2, -c0]),  # sublatices of layer 2
        ('B2', [0, 3*a_cc/2, -c0])
    )

    lat.register_hopping_energies({
        'gamma0': -2.8,  # [eV] intralayer
        'gamma1': -0.4,  # [eV] interlayer
    })

    lat.add_hoppings(
        # layer 1
        ([ 0, 0], 'A1', 'B1', 'gamma0'),
        ([ 0, 1], 'A1', 'B1', 'gamma0'),
        ([-1, 0], 'A1', 'B1', 'gamma0'),
        # layer 2
        ([ 0, 0], 'A2', 'B2', 'gamma0'),
        ([ 0, 1], 'A2', 'B2', 'gamma0'),
        ([-1, 0], 'A2', 'B2', 'gamma0'),
        # interlayer
        ([ 0,  0], 'B1', 'A2', 'gamma1')
    )

    return lat


lattice = bilayer_graphene()
lattice.plot()
plt.show()