import numpy as np
import pybinding as pb
from math import sqrt
from pybinding.repository.graphene.constants import a


def uniform_strain(c_x, c_y):
    """Produce both the displacement and hopping energy modifier"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = c_x * x
        uy = c_y * y
        uz = 0

        return x + ux, y + uy, z + uz

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        r = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
        r_0 = 0.184*a
        v_pi = 2.7 * np.exp(-(r - a/sqrt(3)) / r_0)
        v_sig = - 0.48 * np.exp(-(r - 0.335) / r_0)
        dz = z1 - z2

        return v_pi * (1 - (dz/r)**2) + v_sig * (dz/r)**2

    return displacement, strained_hopping
