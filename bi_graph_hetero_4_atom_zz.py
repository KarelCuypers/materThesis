import pybinding as pb
import matplotlib
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from pybinding.repository.graphene import a, a_cc, t
import numpy as np
from numpy import sqrt
from functions.export_xyz import export_xyz
from functions.create_lattice import create_lattice
from functions.save_LDOS_xyz import save_LDOS_xyz


def add_one_atomic_layer(name, position, onsite_energy, a1, a2, shape):
    """Add a layer made from single sublattice

    Parameters
    ----------

    name : str
        User friendly name for the sublattice.
    position : list
        Position in xyz coordinates.
    onsite_energy : float
        Onsite energy terms at the sublattice.
    a1, a2 : Tuple[float, float, float]
        Unit cell vectors.
    shape:
        Shape of the structure (unit cell).

    Returns
    -------

    Lattice site positions. Named tuple with x, y, z fields, each a 1D array.
    """

    @pb.site_generator(name, onsite_energy)
    def define_layer():
        lat = pb.Lattice(a1=a1, a2=a2)

        lat.add_sublattices((name, position, onsite_energy))

        model = pb.Model(
            lat,
            shape,
        )

        return model.system.positions

    return define_layer


def strained_lattice(onsite=(0, 0, 0, 0), strain=[0, 0]):
    lat = pb.Lattice(a1=[a * (1+strain[0]), 0],
                     a2=[0, 3*a_cc * (1+strain[1])])

    lat.add_sublattices(
        # layer 1
        ('A1', [0,  -a_cc/2 * (1 + strain[1]),   0], onsite[0]),
        ('B1', [0,   a_cc/2 * (1 + strain[1]),   0], onsite[1]),
    )

    lat.add_aliases(
        # Layer 1
        ('A3', 'A1', [a / 2 * (1+strain[0]), (-a_cc/2 + 3*a_cc/2) * (1 + strain[1]), 0]),
        ('B3', 'B1', [a / 2 * (1+strain[0]), (a_cc/2 + 3*a_cc/2) * (1 + strain[1]), 0]),
        # layer 2
    )
    return lat

# check stacking
# check ldos
# check band structure / coupling
# reden dat AA/AB anders is voor zigzag is dat er andere stackings doorlopen worden
# omdat rooster daar niet steeds matched

n_list = [195]
#n_list = [i for i in range(180, 211)]
x_times = 10
y_times = 600

m = 200  # bottom lattice size

hbar = 4.136 * 10 ** (-15)  # eV*s
t_0 = 2.8  # eV
v_F = (3 * t_0 * a_cc) / (2*hbar)  # nm/s

dos_list = []
strain_list = []
E_list = []
E_dip_list = []

for n in n_list:

    atoms_layer_1 = m * 8 * x_times * y_times
    atoms_layer_2 = n * 8 * x_times * y_times

    path = 'C:/Users/Karel/Desktop/Master_Thesis/pythonProject/hetero_dos_calculations/zigzag/10x600_AB/'
    #path = ''
    name = f'4_atom_supercell_size_{n}_over_{m}_x{x_times}_y{y_times}_hetro_strain_zz_AB' #zigzag

    # c_x = 1 / (n - 1)
    c_x = (m-n)/n
    strain_list.append(c_x * 100)
    print('strain size: ', c_x * 100, '%')
    c_y = 0
    strain = [c_x, c_y]

    # make a graphene lattice and extract unit cell vectors a1 and a2
    lattice_gr = strained_lattice(strain=strain)
    a1_strained, a2_strained = lattice_gr.vectors[0], lattice_gr.vectors[1]
    a1, a2 = strained_lattice().vectors[0], strained_lattice().vectors[1]

    # make a unit cell shape from vectors l1 and l2
    strained_shape = pb.primitive(n, 1)
    shape = pb.primitive(m, 1)

    c0 = 0.335  # [nm] interlayer spacing
    strained_model = pb.Model(
        lattice_gr,
        strained_shape,
        # AA
        #add_one_atomic_layer(name='A2', position=[0, -a_cc/2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        #add_one_atomic_layer(name='B2', position=[0, a_cc/2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        #add_one_atomic_layer(name='A4', position=[a / 2, -a_cc / 2 + 3 * a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        #add_one_atomic_layer(name='B4', position=[a / 2, a_cc / 2 + 3 * a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        # AB
        add_one_atomic_layer(name='A2', position=[0, a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        add_one_atomic_layer(name='B2', position=[0, 3 * a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape),
        add_one_atomic_layer(name='A4', position=[a / 2, a_cc / 2 + 3 * a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2,shape=shape),
        add_one_atomic_layer(name='B4', position=[a / 2, 3 * a_cc / 2 + 3 * a_cc / 2, -c0], onsite_energy=0, a1=a1, a2=a2, shape=shape)
    )

    t_perp = 0.035
    if n == m:
        E = np.NaN
    else:
        #E = 3 * 2*np.pi * hbar * v_F / (sqrt(3) * (n*a*(1+c_x))) *2        #linear super lattice energy dip
        E_dip = (hbar * v_F * 2*np.pi)**2 / (t_perp * (n*a*(1+c_x))**2)   #quadratic super lattice energy dip
        E = np.pi / (3 * a) * c_x * v_F * hbar
    E_list.append(E)
    E_dip_list.append(E_dip)
    print(f'{n}: {E}')

    xyz_name = f"hetro_{n}_over_{m}_coord_zz_AB"
    position = strained_model.system.xyz
    #export_xyz(xyz_name, position, a1*m, a2, np.array([0, 0, 1]), ['A'] * position.shape[0])
    #complete_lattice, hoppings = create_lattice(xyz_name, nnn=True)

    # model for kpm

    '''new_model = pb.Model(complete_lattice,
                         pb.primitive(x_times, y_times),
                         pb.translational_symmetry(a1=x_times*complete_lattice.vectors[0][0],
                                                   a2=y_times*complete_lattice.vectors[1][1])
                         )'''

    #kpm = pb.kpm(new_model)
    #dos = kpm.calc_dos(energy=np.linspace(-5, 5, 10000), broadening=0.01, num_random=4*64)
    #pb.save(dos, f'dos_{name}.pbz')

    def rectangle(width, height):
        x0 = width / 2
        y0 = height / 2
        return pb.Polygon([[x0, y0], [x0, -y0], [-x0, -y0], [-x0, y0]])


    '''shape = rectangle(x_times*complete_lattice.vectors[0][0], y_times*complete_lattice.vectors[1][1])

    #ldos = kpm.calc_ldos(energy=np.linspace(-5, 5, 10000), broadening=0.01, position=[0, 0])

    sp_ldos = kpm.calc_spatial_ldos(energy=np.linspace(-5, 5, 10000), broadening=0.1,
                                    shape=shape, reduced_output=True
                                    )'''
    #pb.save(sp_ldos, f'sp_ldos_{name}.pbz')
    #pb.save(ldos, f'ldos_{name}.pbz')

    # also calc ldos over unit cell and show graphically
    # stacking regime calc

    dos = pb.load(f'{path}dos_{name}.pbz')
    show_dos = dos.data / (atoms_layer_1 + atoms_layer_2)
    dos_list.append(show_dos)

strain_index = strain_list
x = strain_index
y = dos.variable
X,Y = np.meshgrid(x, y)
Z = np.array(dos_list).T

matplotlib.rcParams.update({'font.size': 12})
cm = 1 / 2.54
path = 'C:/Users/Karel/Desktop/Master_Thesis/hetero_dos'

#n = n_list[0]

plt.figure()
plt.title(f'Zigzag AB {n} low energy')
plt.plot(dos.variable, dos.data)
plt.xlim(-1, 1)
plt.ylim(0, 0.3*10**6)
plt.axvline(E, color='y')
plt.axvline(E_dip, color='r')
plt.show()

#plt.figure(figsize=(10.5 * cm, 8 * cm), dpi=600)
'''plt.figure()
plt.title('Zigzag AB full spectrum')
plt.pcolormesh(X, Y, Z, cmap='RdYlBu_r')
plt.plot(x, E_list, c='r')
plt.ylim(-5, 5)
plt.xlabel('Strain %')
plt.ylabel('Energy (eV)')
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.savefig(f'{path}/zigzag_full_AB_no_line.png')
plt.show()'''

'''Z = np.array(dos_list).T
cut_off = 0.02
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        if Z[i, j] > cut_off:
            Z[i, j] = np.NaN
#plt.figure(figsize=(10.5 * cm, 8 * cm), dpi=600)
plt.figure()
plt.title('Zigzag AB low E spectrum')
plt.pcolormesh(X, Y, Z, cmap='RdYlBu_r')
plt.ylim(-1, 1)
plt.plot(x, E_list, c='r')
plt.plot(x, E_dip_list, c='y')
plt.xlabel('Strain %')
plt.ylabel('Energy (eV)')
plt.subplots_adjust(left=0.2, bottom=0.15)
plt.savefig(f'{path}/zigzag_lowE_AB_no_line.png')
plt.show()'''

