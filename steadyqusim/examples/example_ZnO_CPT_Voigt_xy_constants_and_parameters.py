import numpy as np
from scipy import constants

from steadyqusim.steady_state_quantum_system_solver import smart_frequency_range_choice_around_resonances
# defining conversion factors and constants
ev_to_ghz = 241799.0504
J_to_ev = 1/constants.value('electron volt')
J_to_ghz = J_to_ev*ev_to_ghz
muB = constants.value('Bohr magneton')
hb = constants.hbar
kB = constants.value('Boltzmann constant')

# defining experimental conditions
T = 1.9  # temperature in Kelvin
B = 6  # B-field in Tesla

# defining material constants
# defining g-factors
gfac = 1.9  # electron g-factor
gfachole = 0.2  # hole g-factor in Voigt

# defining relaxation rates (Do 1/relaxation_time. This will be in cyclic units! No need to divide or multiply by 2pi)
G31 = 1/1  # radiative relaxation rate from 3 -> 1 in inverse ns (no 2Pi!)
# G32 = 0.01  # radiative relaxation rate from 3 -> 2
G32 = 1/1  # radiative relaxation rate from 3 -> 2
# G41 = 0.01  # radiative relaxation rate from 4 -> 1
G41 = 1/1  # radiative relaxation rate from 4 -> 1
G42 = 1/1  # radiative relaxation rate from 4 -> 2
g3 = 0  # additional dephasing between 3-1 and 3-2
g4 = g3  # additional dephasing between 4-1 and 4-2
g2 = 1/170  # spin dephasing between 2-1
G21 = 1/5.5E6  # spin relaxation rate from 2 -> 1, in inverse ns
G12 = G21 * np.exp(-gfac * muB * B / (kB * T))  # spin relaxation rate from 1 -> 2
# print(G21, G12)

# Make Lindbladian using a collapse operator (as per qutip) as a list of collapse operators
temparray = np.zeros((4, 4), dtype=np.complex128)
c_ops = [temparray, temparray, temparray, temparray, temparray, temparray, temparray, temparray, temparray]
# c_ops[0] is collapse operator for population from the excited 3 to ground 1
c_ops[0] = np.sqrt(G31) * np.array([[0. + 0j, 0, 1. + 0j, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# c_ops[1] is collapse operator for population from the excited 3 to ground 2
c_ops[1] = np.sqrt(G32) * np.array([[0. + 0j, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# c_ops[2] is the collapse operator for pure dephasing of of excited state 3
c_ops[2] = np.sqrt(g3) * np.array([[0. + 0j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

# c_ops[3] is the collapse operator for population relaxation from 2 to 1
c_ops[3] = np.sqrt(G21) * np.array([[0. + 0j, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# c_ops[4] is the collapse operator for population relaxation from 1 to 2
c_ops[4] = np.sqrt(G12) * np.array([[0. + 0j, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# c_ops[5] is the collapse operator for pure dephasing of state 2
c_ops[5] = np.sqrt(g2) * np.array([[0. + 0j, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# c_ops[6] is the collapse operator for population from the excited 4 to ground 1
c_ops[6] = np.sqrt(G41) * np.array([[0. + 0j, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

# c_ops[7] is the collapse operator for population from the excited 4 to ground 2
c_ops[7] = np.sqrt(G42) * np.array([[0. + 0j, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])

# c_ops[8] is the collapse operator for pure dephasing of of excited state 4
c_ops[8] = np.sqrt(g4) * np.array([[0. + 0j, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

# Defining the Rabi frequency
Om13 = 0.05  # 0.1 * 2 * np.pi  # Rabi frequency for 1-3, units of Om/2pi are GHz
Om23 = 5E-3  # 0.1 * 2 * np.pi  # Rabi frequency for 2-3
Om24 = 5E-3  # Rabi frequency for 2-4

Del13 = -0.5 * 2 * np.pi * 0  # detuning from 2-3 resonance

peak_distance = gfachole*muB*B*J_to_ghz * 2 * np.pi
Del23 = smart_frequency_range_choice_around_resonances([Del13, peak_distance])  # detuning from 2-3 resonance

Del24 = Del23 - peak_distance  # detuning from 2-4 resonance
