import pybinding as pb
from pybinding.repository import graphene
import numpy as np
from numpy import sqrt
from functions.inhomo_uniform_strain import inhomo_uniform_strain
from functions.export_xyz import export_xyz
from make_lattice_bigrhbn import add_one_atomic_layer, unit_cell, a, a_cc, t, c0


def monolayer_graph(onsite=(0, 0, 0, 0), angle=0):

    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rot_3D = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])

    lat = pb.Lattice(
        a1=np.matmul([a/2, a/2 * sqrt(3)], rot),
        a2=np.matmul([-a/2, a/2 * sqrt(3)], rot)
    )

    c0 = 0.335  # [nm] interlayer spacing
    lat.add_sublattices(
        ('A1', np.matmul([0,  -a_cc/2,   0], rot_3D), onsite[0]),
        ('B1', np.matmul([0,   a_cc/2,   0], rot_3D), onsite[1]),
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
    )
    return lat

# make a graphene lattice and extract unit cell vectors a1 and a2
lattice_gr = monolayer_graph()
a1, a2 = lattice_gr.vectors[0], lattice_gr.vectors[1]
# make a unit cell shape from vectors l1 and l2
strained_shape = unit_cell(9*a1, 10*a2)
shape = pb.primitive(10, 10)

c_x = 1/9
c_y = 0

model = pb.Model(
    lattice_gr,
    strained_shape,
    #add_one_atomic_layer(name='A2', position=[0, a_cc/2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
    #add_one_atomic_layer(name='B2', position=[0, (3/2)*a_cc, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
    inhomo_uniform_strain(c_x, c_y)
)

position = model.system.xyz
y = np.array([10*a2[0]+1/9, 10*a2[1], 0])
export_xyz("test_layers", position, 10*a1, 10*a2, np.array([0, 0, 1]), ['c'] * position.shape[0])

#np.sin(np.deg2rad(30))