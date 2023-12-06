import numpy as np
import matplotlib.pyplot as plt
import pybinding as pb


def plot_3d_band(solver, kx_max, ky_max, band_index=1):

    for ky in range(-ky_max, ky_max):

        K1 = [-kx_max, ky]
        K2 = [kx_max, ky]
        bands = solver.calc_bands(K1, K2)
        E[ky] = bands.energy[:, band_index]

        X = []


