import pybinding as pb


def find_bands_test(name, repeat):
    load_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/lattice_files/'
    save_path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/bilayer_homo_strain/band_files/'

    lattice = pb.load(f'{load_path}lattice_{name}.pbz')

    model = pb.Model(lattice,
                     pb.primitive(repeat, repeat),
                     pb.translational_symmetry(a1=repeat*lattice.vectors[0][0], a2=repeat*lattice.vectors[1][1])
                     )
    solver = pb.solver.arpack(model, k=20)

    # dispersion/band structure 2D/3D
    bands = solver.calc_bands(3, -3, step=0.04)
    #bands = solver.calc_wavefunction(3, -3, step=0.04).bands_disentangled

    #pb.save(bands, f'{save_path}bands_{name}')
    print('Bands found')
