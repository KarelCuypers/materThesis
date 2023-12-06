import pybinding as pb
import numpy as np
import matplotlib.pyplot as plt
from pybinding.repository import graphene
from math import pi
from pybinding.constants import phi0


def triaxial_strain(c):
    """Strain-induced displacement and hopping energy modification"""
    @pb.site_position_modifier
    def displacement(x, y, z):
        ux = 2*c * x*y
        uy = c * (x**2 - y**2)
        return x + ux, y + uy, z

    @pb.hopping_energy_modifier
    def strained_hopping(energy, x1, y1, z1, x2, y2, z2):
        l = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        w = l / graphene.a_cc - 1
        return energy * np.exp(-3.37 * w)

    return displacement, strained_hopping


def constant_magnetic_field(B):
    @pb.hopping_energy_modifier
    def function(energy, x1, y1, x2, y2):
        # the midpoint between two sites
        y = 0.5 * (y1 + y2)
        # scale from nanometers to meters
        y *= 1e-9

        # vector potential along the x-axis
        A_x = B * y

        # integral of (A * dl) from position 1 to position 2
        peierls = A_x * (x1 - x2)
        # scale from nanometers to meters (because of x1 and x2)
        peierls *= 1e-9

        # the Peierls substitution
        return energy * np.exp(1j * 2*pi/phi0 * peierls)
    return function


# test file to compare effect of a magnetic field and effect of strain in a hexagon
# I don't know expectedly what strain leads to what level of field soo the strength off both effect differs

hexagon = pb.regular_polygon(num_sides=6, radius=50, angle=np.pi/6)

model = pb.Model(graphene.monolayer(), hexagon)
model_field = pb.Model(graphene.monolayer(), hexagon, constant_magnetic_field(200))
model_strain = pb.Model(graphene.monolayer(), hexagon, triaxial_strain(0.1/50))

kpm = pb.kpm(model)
kpm_field = pb.kpm(model_field)
kpm_strain = pb.kpm(model_strain)

ldos = kpm.calc_ldos(energy=np.linspace(-1, 1, 500), broadening=0.015, position=[0, 0])
dos = kpm.calc_dos(energy=np.linspace(-1, 1, 500), broadening=0.06, num_random=16)
ldos_field = kpm_field.calc_ldos(energy=np.linspace(-1, 1, 500), broadening=0.015, position=[0, 0])
dos_field = kpm_field.calc_dos(energy=np.linspace(-1, 1, 500), broadening=0.06, num_random=16)
ldos_strain = kpm_strain.calc_ldos(energy=np.linspace(-1, 1, 500), broadening=0.015, position=[0, 0])
dos_strain = kpm_strain.calc_dos(energy=np.linspace(-1, 1, 500), broadening=0.06, num_random=16)

# on figure top row is de density of states for the magnetic field
# bottom layer is for the strained graphene
# both clearly result in landau level splitting


fig = plt.figure()
ax1 = fig.add_subplot(321)
ldos.plot()
ax2 = fig.add_subplot(322)
dos.plot()
ax3 = fig.add_subplot(323)
ldos_field.plot()
ax4 = fig.add_subplot(324)
dos_field.plot()
ax5 = fig.add_subplot(325)
ldos_strain.plot()
ax6 = fig.add_subplot(326)
dos_strain.plot()
ax1.title.set_text('LDOS no strain or field')
ax2.title.set_text('DOS no strain or field')
ax3.title.set_text('LDOS filed')
ax4.title.set_text('DOS field')
ax5.title.set_text('LDOS strain')
ax6.title.set_text('DOS strain')
plt.show()