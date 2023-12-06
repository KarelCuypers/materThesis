import pybinding as pb
import numpy as np
from pybinding.repository import graphene
from functions.bilayer_4atom import bilayer_4atom
from functions.unit_cell import unit_cell
from functions.uniform_strain import uniform_strain
from functions.four_atom_gating_term import four_atom_gating_term
from functions.export_xyz import export_xyz

c_x = 0.05  #zz
c_y = 0  #ac

a1, a2 = bilayer_4atom().vectors[0] * (1 + c_x), bilayer_4atom().vectors[1] * (1 + c_y)
a = graphene.a_cc * np.sqrt(3)
a_cc = graphene.a_cc

times_l1 = 10
times_l2 = 10

l1_size = times_l1 * np.linalg.norm(a1)

l2_size = times_l2 * np.linalg.norm(a2)

strained_model = pb.Model(
    bilayer_4atom(),
    unit_cell(l1=2 * times_l1 * a1, l2=2 * times_l1 * a2),
    pb.translational_symmetry(a1=l1_size, a2=l2_size),  # always needs some overlap with the rectangle
    uniform_strain(c_x, c_y),
    four_atom_gating_term(0.5)
)

position = strained_model.system.xyz

path = '/bilayer_homo_strain/xyz_files/'
name = 'uniform_strain_xyz_test'

export_xyz(f'{path}{name}', position, l1_size * a1 / np.linalg.norm(a1), l2_size * a2 / np.linalg.norm(a2),
           np.array([0, 0, 1]), ['A1'] * position.shape[0])
