import pybinding as pb
import numpy as np
import copy
import re

from scipy.spatial import cKDTree
from math import pi, sqrt
from pybinding.repository.graphene import a, a_cc


c0 = 0.335  # [nm] graphene interlayer spacing
# Constants and hopping values based on Moon's paper
t = -2.7
V0_pi = t  # [eV] graphene NN intralayer hopping
V0_sigma = 0.48  # [eV] graphene NN interlayer hopping
rc = 0.614  # [nm] hopping fitting parameter
lc = 0.0265  # [nm] hopping fitting parameter
q_sigma = c0 * 22.2  # [nm] hopping fitting parameter
q_pi = a_cc * 22.2  # [nm] hopping fitting parameter
r0 = 0.184 * a  # [nm] hopping fitting parameter


def find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter):
    """ Find indexes for interlayer neighbours
    positions : np.ndarray
        Position of sites.
    positions_to : np.ndarray
        Position of sites that are being connected.
    l1, l2 : np.array, np.array
        Unit cell vectors.
    d_min_intra, d_max_intra : float, float
        Min and max distance for interlayer neighbour.

    Returns
    -------
    np.array, np.array
        Index of hopping from , index of hopping to.
    """

    z = positions[:, 2]
    layer1 = z < c0 / 2
    layer2 = z > c0 / 2
    
    row1, col1 = [], []


    if np.logical_and(np.any(layer1), np.any(layer2)):
        kdtree1 = cKDTree(positions[layer1])
        kdtree2 = cKDTree(positions_pbc[layer2])

        abs_idx1 = np.flatnonzero(layer1)
        abs_idx2 = np.flatnonzero(layer2)

        coo = kdtree1.sparse_distance_matrix(kdtree2, d_max_inter, output_type='coo_matrix')

        idx = coo.data > d_min_inter
        row1, col1 = abs_idx1[coo.row[idx]], abs_idx2[coo.col[idx]]


    return np.array(row1).astype(int), np.array(col1).astype(int)


def find_row_and_col_intra(positions, positions_to, d_min_intra, d_max_intra):
    """ Find indexes for intralayer neighbours
    positions : np.ndarray
        Position of sites.
    positions_to : np.ndarray
        Position of sites that are being connected.
    l1, l2 : np.array, np.array
        Unit cell vectors.
    d_min_intra, d_max_intra : float, float
        Min and max distance for intralayer neighbour.

    Returns
    -------
    np.array, np.array
        Index of hopping from , index of hopping to.
    """

    def find_neigh_in_layer(layer):

        kdtree1 = cKDTree(data=positions[layer])
        kdtree2 = cKDTree(data=positions_to[layer])

        coo = kdtree1.sparse_distance_matrix(kdtree2, d_max_intra, output_type='coo_matrix')
        abs_idx = np.flatnonzero(layer)

        idx = coo.data > d_min_intra
        row, col = abs_idx[coo.row[idx]], abs_idx[coo.col[idx]]
        return row, col

    z = positions[:, 2]
    layer1 = z < c0 / 2
    layer2 =  z > c0 / 2
    
    row1, col1, row2, col2 = [], [], [], []
    
    if np.any(layer1):
        row1, col1 = find_neigh_in_layer(layer1)

    if np.any(layer2):
        row2, col2 = find_neigh_in_layer(layer2)
    

    row = np.concatenate((row1, row2))
    col = np.concatenate((col1, col2))

    return row.astype(int), col.astype(int)


def periodic_neighbors(positions, l1, l2, d_min_inter, d_max_inter, d_min_intra, d_max_intra):
    """Find periodic neighbours based on the image algorithm
    Make images and find neighbours.
    No need to do +L and -L. One is enough, the other is automatically added to the lattice.

    positions : np.ndarray
        Position of sites.
    l1, l2 : np.array, np.array
        Unit cell vectors.
    d_min_intra, d_max_intra, d_min_inter, d_max_inter : float, float, float, float
        Min and max distance for intra and interlayer neighbours.

    Returns
    -------
    np.array, np.array, np.ndarray
        Index of hopping from , index of hopping to, unit cell reference index for all images, PBC.
    """

     # +l1
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] += l1[0]
    positions_pbc[:, 1] += l1[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row1, col1 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell1 = np.column_stack((np.ones(row1.shape[0], dtype=int), np.zeros(row1.shape[0], dtype=int)))

    # -l1
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] -= l1[0]
    positions_pbc[:, 1] -= l1[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row2, col2 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell2 = np.column_stack((-np.ones(row2.shape[0], dtype=int), np.zeros(row2.shape[0], dtype=int)))

    # +l2
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] += l2[0]
    positions_pbc[:, 1] += l2[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row3, col3 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell3 = np.column_stack((np.zeros(row3.shape[0], dtype=int), np.ones(row3.shape[0], dtype=int)))

    # -l2
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] -= l2[0]
    positions_pbc[:, 1] -= l2[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row4, col4 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell4 = np.column_stack((np.zeros(row4.shape[0], dtype=int), -np.ones(row4.shape[0], dtype=int)))

    # +l1 + l2
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] += l1[0] + l2[0]
    positions_pbc[:, 1] += l1[1] + l2[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row5, col5 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell5 = np.column_stack((np.ones(row5.shape[0], dtype=int), np.ones(row5.shape[0], dtype=int)))

    # -l1 - l2
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] -= l1[0] + l2[0]
    positions_pbc[:, 1] -= l1[1] + l2[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row6, col6 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell6 = np.column_stack((-np.ones(row6.shape[0], dtype=int), -np.ones(row6.shape[0], dtype=int)))

    # +l1 - l2
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] += l1[0] - l2[0]
    positions_pbc[:, 1] += l1[1] - l2[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row7, col7 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell7 = np.column_stack((np.ones(row7.shape[0], dtype=int), -np.ones(row7.shape[0], dtype=int)))

    # -l1 + l2
    positions_pbc = copy.deepcopy(positions)
    positions_pbc[:, 0] -= l1[0] - l2[0]
    positions_pbc[:, 1] -= l1[1] - l2[1]
    row_intra, col_intra = find_row_and_col_intra(positions, positions_pbc, d_min_intra, d_max_intra)
    row_inter, col_inter = find_row_and_col_inter(positions, positions_pbc, d_min_inter, d_max_inter)
    row8, col8 = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell8 = np.column_stack((-np.ones(row8.shape[0], dtype=int), np.ones(row8.shape[0], dtype=int)))

    row = np.concatenate((row1, row2, row3, row4, row5, row6, row7, row8))
    col = np.concatenate((col1, col2, col3, col4, col5, col6, col7, col8))
    pbc_cell = np.concatenate((pbc_cell1, pbc_cell2, pbc_cell3, pbc_cell4, pbc_cell5, pbc_cell6, pbc_cell7, pbc_cell8))

    return row, col, pbc_cell


def unit_cell_neighbours(positions, d_min_intra, d_max_intra, d_min_inter, d_max_inter):
    """Find the neighbours inside the unit cell.

    Parameters
    ----------

    positions : np.ndarray
        Position of sites.
    d_min_intra, d_max_intra, d_min_inter, d_max_inter : float, float, float, float
        Min and max distance for intra and interlayer neighbours.

    Returns
    -------
    np.array, np.array, np.ndarray
        Index of hopping from , index of hopping to, unit cell reference index
    """
    positions_to = copy.deepcopy(positions)

    # find intralayer hopping
    row_intra, col_intra = find_row_and_col_intra(positions, positions_to, d_min_intra, d_max_intra)
    # find interlayer hopping
    row_inter, col_inter = find_row_and_col_inter(positions, positions_to, d_min_inter, d_max_inter)

    row, col = np.append(row_intra, row_inter), np.append(col_intra, col_inter)
    pbc_cell = np.column_stack((np.zeros(row.shape[0], dtype=int), np.zeros(row.shape[0], dtype=int)))

    return row, col, pbc_cell


def load_ovito_lattice(name):
    """Load a lattice from a Ovito .xyz file. At the moment, works for files that have following format
       Type_id X_coord Y_coord Z_coord Data
       The point needs to have all 3 coordinates, and the approach is not general.

    Parameters
    ----------

    name : str
        Name of the xyz file

    Return
    ----------

        x, y, z coordinates, atom types and l1, l2, l3 unit cell vectors
    """
    load_f = open(name, 'r')

    # read the atom number
    num_atoms = load_f.readline()

    # read the lattice vectors
    vectors = load_f.readline()
    vec = re.findall(r"[-+]?\d*\.*\d+", vectors)
    vec = list(map(float, vec))

    space_size = int(sqrt(len(vec)))

    _l1, _l2, _l3 = vec[0:space_size], vec[space_size:2 * space_size], vec[2 * space_size:3 * space_size]

    _atom_type = []
    _x_coord = []
    _y_coord = []
    _z_coord = []
    for line in load_f:
        atom = []
        for u in line.split():
            u = u.strip('\'')
            atom.append(u)
        _atom_type.append(atom[0])
        _x_coord.append(float(atom[1]) * 0.1)
        _y_coord.append(float(atom[2]) * 0.1)
        _z_coord.append(float(atom[3]) * 0.1)

    _x_coord = np.asarray(_x_coord)
    _y_coord = np.asarray(_y_coord)
    _z_coord = np.asarray(_z_coord)
    _atom_type = np.asarray(_atom_type)

    return _x_coord, _y_coord, _z_coord, _atom_type, _l1, _l2, _l3


def make_lattice(a1, a2, atom_type, positions, different_atoms, onsite_potential):
    """Import the loaded atoms to the lattice

    Parameters
    ----------

    a1, a2 : np.array, np.array
        Unit cell vectors
    atom_type : np.ndarray
        Sublattice labels.
    positions : np.ndarray
        Position of sites.
    different_atoms : list[str]
        Atom types loaded from XYZ
    onsite_potential : list[float]
        Onsite energy at each sublattice.

    Return
    ----------
        pb.Lattice lattice defined without the hopping terms

    """
    lat = pb.Lattice(a1=a1, a2=a2)
    list_atoms = []
    count = np.zeros(len(different_atoms), dtype=int)

    for [i, atom] in enumerate(positions):
        if atom_type[i] in different_atoms:
            j = different_atoms.index(atom_type[i])
            list_atoms.append((different_atoms[j] + str(count[j]), atom, onsite_potential[i]))
            count[j] += 1
    lat.add_sublattices(*list_atoms)

    return lat


def make_complete_lattice(lattice, row, col, cell, hoppings):
    """Import the loaded atoms to the lattice

    Parameters
    ----------

    lattice : pb.Lattice
        Reference lattice without hopping.
    row : np.ndarray
        Hopping from index.
    col : np.ndarray
        Hopping to index.
    cell : np.array
        List of reference unit cells for each hopping.
    hoppings : np.array
        Hopping terms.

    Return
    ----------
    pb.Lattice lattice defined with the hopping terms

    """

    lat = pb.Lattice(a1=lattice.vectors[0][0:2], a2=lattice.vectors[1][0:2])

    sub_alias_id = {}
    # add each sublattice
    for key, value in lattice.sublattices.items():
        sub_alias_id[value.alias_id] = key
        lat.add_one_sublattice(
            key, value.position, value.energy
        )

    # add each hopping term
    for unit_cell, hopping, r, c in zip(cell, hoppings, row, col):
        lat.add_one_hopping(
            unit_cell, sub_alias_id[r], sub_alias_id[c], hopping
        )
    lat.min_neighbors = 1
    return lat


def calculate_hoppings(row, col, positions, cell, l1, l2):
    """Import the loaded atoms to the lattice

    Parameters
    ----------

    row : np.ndarray
        Hopping from index.
    col : np.ndarray
        Hopping to index.
    positions : np.ndarray
        Position of sites.
    cell : np.array
        List of reference unit cells for each hopping.
    l1, l2 : np.array
        Unit cell vectors.
    Return
    ----------
        List of hopping terms.

    """

    x1, y1, z1 = positions[row, 0], positions[row, 1], positions[row, 2]

    x2, y2, z2 = positions[col, 0] + cell[:, 0] * l1[0] + cell[:, 1] * l2[0], \
                 positions[col, 1] + cell[:, 0] * l1[1] + cell[:, 1] * l2[1], \
                 positions[col, 2]

    dx, dy, dz = x1 - x2, y1 - y2, z1 - z2

    d = np.sqrt(np.square(dx) + np.square(dy) + np.square(dz))
    d = np.squeeze(np.asarray(d))
    dz = np.squeeze(np.asarray(dz))

    f_c = 1.0 / (1.0 + np.exp((d - rc) / lc))
    v_pi = V0_pi * np.exp(q_pi * (1.0 - d / a_cc)) * f_c
    v_sigma = V0_sigma * np.exp(q_sigma * (1.0 - d / c0)) * f_c

    hopping = v_pi * (1 - (dz / d) ** 2) + v_sigma * (dz / d) ** 2

    return hopping
