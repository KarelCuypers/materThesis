import os

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

import pybinding as pb
import matplotlib.pyplot as plt
import numpy as np
import sys

from numpy import pi, sqrt
from pybinding.repository import graphee
from pybinding.repository.graphene import a, a_cc, t
from concurrent import futures


load_path = '/scratch/antwerpen/209/vsc20947/lattice_files/'
save_path = '/scratch/antwerpen/209/vsc20947/band_files/'
name = 'uniform_strain_xyz_test.xyz'
#name = str(sys.argv[1])

complete_lattice = pb.load(f'{load_path}lattice_{name}')
savename = 'bands_' + name
savename_wfc = 'wfc_' + name

numbands = 30
kstep = 0.001
#sig = float(sys.argv[2])
sig = 0

c0 = 0.335

l1, l2 = complete_lattice.vectors
points = complete_lattice.brillouin_zone()
K1 = points[0]
K2 = points[1]
M = (K1 + K2)/2
gamma = [0, 0]

positions = pb.Model(complete_lattice).system.xyz
z_coord = positions[:, 2]


def get_mean_heights(z):
    
    
    def get_num_layers(z):
        
        minimum = np.min(z)
        iterator = 0    
        while np.logical_and(z > minimum -c0/2 + iterator * c0, z < minimum + c0/2 + iterator * c0).any():
            iterator += 1

        return iterator
    
    
    num_layers = get_num_layers(z)
    mean_heights = []
    minimum = np.min(z)

    for i in range(num_layers):
        layer = np.logical_and(z > minimum -c0/2 + i * c0, z < minimum + c0/2 + i * c0)
        mean_heights.append(np.mean(z[layer]))
    
    height_diff = np.zeros((np.size(mean_heights)-1,1))
    for idx, height in enumerate(height_diff):
        height_diff[idx] = mean_heights[idx+1] - mean_heights[idx]
    
    return mean_heights


def apply_E_field(V_field, mean_heights):
    
    @pb.onsite_energy_modifier
    def potential(energy, z):
        
        minimum = mean_heights[0]
        bottom_layer_V = -(mean_heights[-1] - minimum) * V_field / 2
        
        for idx, i in enumerate(mean_heights):
            current_layer = np.logical_and(z > minimum -c0/2 + idx * c0, z < minimum + c0/2 + idx * c0)
            energy[np.flatnonzero(current_layer)] += bottom_layer_V + (mean_heights[idx] - minimum) * V_field
            
        return energy
    
    return potential


def determine_bands(ki):

    # model must include symmetry again

    model = pb.Model(complete_lattice,
                     pb.translational_symmetry())

    solver = pb.solver.arpack(model, k=numbands, sigma=sig)
    solver.set_wave_vector(ki)

    return solver.eigenvalues, solver.eigenvectors.T

mean_heights = get_mean_heights(z_coord)
k_path = pb.make_path(gamma, K1, K2, gamma, step=kstep)
bands, wfc = [], []
idx = np.arange(0, len(k_path))

with futures.ProcessPoolExecutor() as executor:
    for kp, i, result in zip(k_path, idx, executor.map(determine_bands, k_path)):
        bands.append(result[0])
        #wfc.append(result[1])

bands_object = pb.Bands(k_path, bands)
#wfc_object = np.array(wfc)
#bands_object = pb.Wavefunction(bands_object, wfc_object).bands_disentangled
#pb.save(wfc_object, f"{savename_wfc}_{str(int(gating*1000))}_mev_r")
pb.save(bands_object, f"{save_path}{savename}.pbz")
