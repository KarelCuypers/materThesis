import pybinding as pb


def four_atom_gating_term(delta):
    @pb.onsite_energy_modifier
    def potential(energy, sub_id):
        #energy[sub_id == 'A2' or sub_id == 'B2' or sub_id == 'A4' or sub_id == 'B4'] += delta
        #energy[sub_id == 'A1' or sub_id == 'B1' or sub_id == 'A3' or sub_id == 'B3'] -= delta
        energy[sub_id == 'A2' or sub_id == 'B2'] += delta
        energy[sub_id == 'A1' or sub_id == 'B1'] -= delta
        return energy
    return potential
