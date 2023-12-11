from functions.calculate_hoppings import *
from pathlib import Path

load_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/xyz_files/'
save_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/lattice_files/'

# # load coordinates, atom types and lattice vectors
name = 'uniform_strain_xyz_test.xyz'
x_coord, y_coord, z_coord, atom_type, l1, l2, l3 = load_ovito_lattice(f'{load_path}{name}')
l1 = np.array(l1[0:2]) / 10.
l2 = np.array(l2[0:2]) / 10.
positions = np.column_stack((x_coord, y_coord, z_coord))

# name for different sublattices, still every added atom to the model needs to have a different sublattice ID,
# and this is where the sublattice info is lost
different_atoms = ['A1']

# Setting the onsite potentials, for now all are zero until there is something to add
onsite_potential = np.zeros((positions.shape[0],))

# min and max interlayer radius for neighbour search
d_min_intra, d_max_intra = 0.5 * a_cc, 1.4 * a_cc
# min and max intralayer radius for neighbour search
d_min_inter, d_max_inter = 0.5 * c0, 1.4 * c0

# make a lattice without hopping
lattice = make_lattice(l1, l2, atom_type, positions, different_atoms, onsite_potential)

# find periodic hopping
row_pbc, col_pbc, cell_pbc = periodic_neighbors(positions, l1=l1, l2=l2,
                                                d_min_inter=d_min_inter, d_max_inter=d_max_inter,
                                                d_min_intra=d_min_intra, d_max_intra=d_max_intra)

# find hopping inside the unit cell
row_unit, col_unit, cell_unit = unit_cell_neighbours(positions, d_min_inter=d_min_inter, d_max_inter=d_max_inter,
                                                     d_min_intra=d_min_intra, d_max_intra=d_max_intra)

# for hopping i, j (i < j) conjugated hopping (j, i) is added automatically. (i, i) needs to be added as a onsite energy
# not a hopping.
row_unit, col_unit, cell_unit = row_unit[row_unit < col_unit], col_unit[row_unit < col_unit], \
                                cell_unit[row_unit < col_unit]

row_pbc, col_pbc, cell_pbc = row_pbc[row_pbc < col_pbc], col_pbc[row_pbc < col_pbc], \
                                cell_pbc[row_pbc < col_pbc]

row = np.concatenate((row_pbc, row_unit))
col = np.concatenate((col_pbc, col_unit))
cell = np.concatenate((cell_pbc, cell_unit))

# calculate the hoppings based on the distance
hoppings = calculate_hoppings(row, col, positions, cell, l1, l2)
# make a full pb.Lattice with sites and hoppings
complete_lattice = make_complete_lattice(lattice, row, col, cell, hoppings)
pb.save(complete_lattice, f'{save_path}lattice_{name}')
