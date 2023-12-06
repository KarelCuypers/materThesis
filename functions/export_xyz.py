import numpy as np
import pybinding as pb
import re
import copy
import warnings
from pybinding.repository import graphene


def export_xyz(name, positions, l1, l2, l3, types, origin = np.array([0,0,0]), *variable):
    """Export the xyz file from a Pybinding model with the possibility to add an extra on-site variable.

    Parameters
    ----------
    name : str
        Name of the xyz file
    positions: np.array
        Positions of the atomic sites, has to be an array with each row [x,y,z] corresponding to an atomic position. 
    types: list[str]
        User friendly names for the sublattices.
    l1, l2, l3: np.array
        The lattice vectors.
    types: list[str]
        List with the type of every atom.
    origin: np.array
        Origin of the unit cell. Default is set to np.array([0,0,0]).
    *variable: float
        Additional variable defined on each atomic site. Creates new column in the file.
    """
                
       
    num_sites = positions.shape[0]
    x = positions[:,0] * 10
    y = positions[:,1] * 10
    z = positions[:,2] * 10
    
    
    variable_used = False
    if len(variable) > 0:
        variable_used = True


    f = open(name + '.xyz', 'w')
    f.write('{0:1d}\n'.format(num_sites))
    f.write(
        'Lattice=" {R1x:.15f} {R1y:.15f} {R1z:.15f} {R2x:.15f} {R2y:.15f} {R2z:.15f} {R3x:.15f} {R3y:.15f} {R3z:.15f}"'.format(R1x=10 * l1[0], R1y=10 * l1[1], R1z=10 * l1[2],
                      R2x=10 * l2[0], R2y=10 * l2[1], R2z=10 * l2[2],
                      R3x=10 * l3[0],     R3y=10 * l3[1],     R3z=10 * l3[2])
        + ' Origin=" {Ox:.10f} {Oy:.10f} {Oz:.10f}''"\n'.format(Ox =10 * origin[0], Oy =10 * origin[1], Oz =10 * origin[2]))


    for idx, (lc, xc, yz, zc) in enumerate(zip(types, x, y, z)):
        f.write(f'{lc} {xc} {yz} {zc}')
        if variable_used:
            for var in variable:
                f.write(f' {var[idx]}')
        f.write('\n')
    f.close()


if __name__ == "__main__":
    armchair_model = pb.Model(
        graphene.monolayer(gamma3=False),
        pb.translational_symmetry(a1=True, a2=False),
        pb.rectangle(2)
    )

    position = armchair_model.system.xyz

    export_xyz("testxyz", position, armchair_model.lattice.vectors[0]*10, armchair_model.lattice.vectors[1]*10,
           np.array([0,0, 100]), ['c']*position.shape[0])
