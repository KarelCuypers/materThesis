import os
import numpy as np


def save_LDOS_xyz(filepath, Evals, coord, types, ldos, modulus_data=5):
    """
    Saves LDOS data into an xyz dump file where every step is the LDOS at a certain energy.
    If a file with the same name as 'filepath.xyz' already exists it will be overwritten.

    Parameters
    ----------
    filepath : str
        Name of the xyz dump file.
    Evals : np.array
        Energy values at each step.
    coord : np.array
        Array where each element is the position [x,y,z] of a site.
    types : np.array
        Array or list containing the atom types.
    ldos : np.array
        n x m array containing the ldos at each position (n positions) for each energy value (m values).
    modulus_data: int
        Only exports every n'th step where n = modulus_data, 1 exports all steps.'
    Returns
    -------
    None.

    """

    if os.path.exists(f'{filepath}.xyz'):
        os.remove(f'{filepath}.xyz')

    for idx, i in enumerate(Evals):
        if idx % modulus_data == 1:
            f = open(filepath, 'a')
            f.write('{0:1d}\n'.format(np.size(coord[:, 0])))
            f.write('Energy:' + str(i) + '\n')

            for lc, xc, yz, zc, Ec in zip(types, coord[:, 0], coord[:, 1], coord[:, 2], ldos[idx, :]):
                f.write('{0} {1:.10f} {2:.10f} {3:.10f} {4:.10f}\n'.format(lc, 10 * xc, 10 * yz, 10 * zc, Ec))

            f.close()
