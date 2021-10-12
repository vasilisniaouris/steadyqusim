
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
import pandas as pd
from scipy.optimize import curve_fit
from steadyqusim.steady_state_quantum_system_solver import smart_frequency_range_choice_around_resonances
import io, pkgutil

import steadyqusim.steady_state_quantum_system_solver as ssqss
from steadyqusim.hamiltonians.ZnO_shallow_donor_hamiltonians import \
    hamiltonian_four_level_degenerate_excited_state as hamiltonian

# defining conversion factors and constants
ev_to_ghz = 241799.0504
J_to_ev = 1/constants.value('electron volt')
J_to_ghz = J_to_ev*ev_to_ghz
muB = constants.value('Bohr magneton')
hb = constants.hbar
kB = constants.value('Boltzmann constant')

# defining experimental conditions
T = 5.4  # temperature in Kelvin
B = 7  # B-field in Tesla

# defining material constants
# defining g-factors
gfac = 1.97  # electron g-factor


def expfieldtotemp(magnetic_field, temperature):
    return np.exp(gfac * muB * magnetic_field/(kB * temperature))


T1at1p5K = 3E6
T1atT = T1at1p5K/((expfieldtotemp(B, T) + 1)/(expfieldtotemp(B, T) - 1)*(expfieldtotemp(B, 1.5) - 1)/(expfieldtotemp(B, 1.5) + 1))
T1 = 3E6  # spin relaxation time in ns
# defining relaxation rates (Do 1/relaxation_time. This will be in cyclic units! No need to divide or multiply by 2pi)
G12 = (expfieldtotemp(B, T) - 1)/(expfieldtotemp(B, T) + 1)/T1 # spin relaxation rate from 1 -> 2
G21 = 1/T1 - G12  # spin relaxation rate from 2 -> 1, in inverse ns
# G12 = G21 * np.exp(-gfac * muB * B / (kB * T))

Delta_gr = gfac * muB * B * J_to_ghz * 2 * np.pi


def get_c_ops(G31, G32, g3, g2):

    G41 = G32
    G42 = G31
    g4 = g3

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

    return np.array(c_ops)


def objective_func(x, g3_1, g3_2, g3_3, g3_4, g2, Om2, Om1, phi, center, norm,
                   lin_a_1, lin_a_2, lin_a_3, lin_a_4, lin_b_1, lin_b_2, lin_b_3, lin_b_4):
    x, powers_fraction = x
    the_four_fracitons = np.unique(powers_fraction)

    Del1 = 811489.4 - center

    # c_ops_1 = get_c_ops(0.74, 0.74, g3 * the_four_fracitons[0] ** (0.5), g2)
    # c_ops_2 = get_c_ops(0.74, 0.74, g3 * the_four_fracitons[1] ** (0.5), g2)
    # c_ops_3 = get_c_ops(0.74, 0.74, g3 * the_four_fracitons[2] ** (0.5), g2)
    # c_ops_4 = get_c_ops(0.74, 0.74, g3 * the_four_fracitons[3] ** (0.5), g2)

    c_ops_1 = get_c_ops(0.74, 0.74, g3_1, g2)
    c_ops_2 = get_c_ops(0.74, 0.74, g3_2, g2)
    c_ops_3 = get_c_ops(0.74, 0.74, g3_3, g2)
    c_ops_4 = get_c_ops(0.74, 0.74, g3_4, g2)

    c_ops = [c_ops_1, c_ops_2, c_ops_3, c_ops_4]
    norms = [norm, norm, norm, norm]
    lin_as = [lin_a_1, lin_a_2, lin_a_3, lin_a_4]
    lin_bs = [lin_b_1, lin_b_2, lin_b_3, lin_b_4]

    nuclear_spin_spread = 0.050*(np.linspace(-5, 4, 10)+0.5)
    Del1 = (Del1 - nuclear_spin_spread)*2*np.pi

    if isinstance(x, float):
        h = []
        for j in range(len(nuclear_spin_spread)):
            h.append(hamiltonian(delta1=Del1[j], delta2=(x-center+nuclear_spin_spread)*2*np.pi, omega1=Om1, omega2=Om2,
                                 phi=phi))
    else:
        Om1 = np.array(powers_fraction) * Om1
        # norm = np.array(powers_fraction) * norm
        h = []
        for j in range(len(nuclear_spin_spread)):
            h.append(np.array([hamiltonian(delta1=Del1[j], delta2=(x[i] - center + nuclear_spin_spread[j]) * 2 * np.pi,
                                           omega1=Om1[i], omega2=Om2, phi=phi)
                               for i in range(len(x))]))

    total_fitted = np.zeros(x.shape)
    for j in range(len(nuclear_spin_spread)):
        fitted_ys = []
        for i, power_fraction in enumerate(np.unique(powers_fraction)):
            indeces = powers_fraction == power_fraction
            partial_x = np.array(x[indeces])
            partial_h = np.array(h[j][indeces])
            partial_solution = ssqss.solve_system(partial_h, c_ops[i])
            partial_excited_states_population = ssqss.retrieve_part_of_solution(partial_solution, [2, 3], [2, 3])
            fitted_ys.append((partial_excited_states_population*norms[i]) + (lin_as[i] * (partial_x-center) + lin_bs[i]))

        all_fitted_y = []
        for fitted_y in fitted_ys:
            all_fitted_y = all_fitted_y + list(fitted_y)

        total_fitted = total_fitted + np.array(all_fitted_y)

    return total_fitted/len(nuclear_spin_spread)/powers_fraction


# ------------------------- Getting files ------------------------------------
powers = [1250, 5000, 20000, 40000]
# powers = [1250, 5000]
# powers = [1250]
data_frames = []
heights = []
for power in powers:
    packaged_data = pkgutil.get_data('steadyqusim.examples.data', f'CPT_{int(power)}.0.csv')
    dataframe = pd.read_csv(io.BytesIO(packaged_data))
    data_frames.append(dataframe)
    data_frames[-1]['power'] = power
    data_frames[-1]['power fraction'] = np.sqrt(power/40000)
    data_frames[-1]['Frequency (GHz)'] = data_frames[-1]['Lsr: Energy (eV)']*ev_to_ghz
    data_frames[-1]['sigma'] = (np.abs(data_frames[-1]['Frequency (GHz)'] - 811489.5))**(1/2)
    heights.append(max(data_frames[-1]['Corrected PL'])-min(data_frames[-1]['Corrected PL']))

total_data = pd.concat(data_frames, ignore_index=True)
pd.set_option('display.max_rows', total_data.shape[0]+1)
pd.set_option('display.max_columns', total_data.shape[0]+1)
# print(total_data)
# ------------------ Setting initial parameters -------------------------------
power_reference = 40000
deph_exc = 50
T2star = 20
om2 = 9
om1 = 14
phi0 = 0

del1 = 7.4
cent = 811479

linear_a = {40000: 7, 20000: 7, 5000: 4, 1250: 2.5}
linear_b = {40000: 6300, 20000: 4500, 5000: 2000, 1250: 1300}

kwp2 = {'g3_1': 0, 'g3_2': 60, 'g3_3': 73, 'g3_4': 127,
        'g2': 1/T2star,
        'Om2': om2, 'Om1': om1,
        'phi': phi0,
        'center': cent,
        'norm': 2.08E4,
        'lin_a_1': linear_a[1250], 'lin_a_2': linear_a[5000], 'lin_a_3': linear_a[20000], 'lin_a_4': linear_a[40000],
        'lin_b_1': linear_b[1250], 'lin_b_2': linear_b[5000], 'lin_b_3': linear_b[20000], 'lin_b_4': linear_b[40000]
        }

N = 20000
x_data = np.linspace(811300, 811630, N)
four_data_x = np.array(list(x_data)*len(powers))
four_data_pf = []
for i in range(len(powers)):
    four_data_pf += [np.sort(total_data['power fraction'].unique())[i]]*N


for power in np.sort(total_data['power'].unique()):
    indeces = total_data['power'] == power
    total_data_x = np.array(total_data['Frequency (GHz)'][indeces])
    total_data_y = np.array(total_data['Corrected PL'][indeces])
    total_data_power = np.array(total_data['power'][indeces])
    total_data_power_fraction = np.array(total_data['power fraction'][indeces])
    plt.plot(total_data_x, total_data_y, '.-', label=power)
#     # fit_y = objective_func((x_data, [total_data_power_fraction[0]]*len(x_data)), *p0)
#     # plt.plot(x_data, fit_y, '-', label=power)


# total_data = total_data.loc[(total_data['Frequency (GHz)'] < 811530) & (total_data['Frequency (GHz)'] > 811450)]
total_data_x = np.array(total_data['Frequency (GHz)'])
total_data_y = np.array(total_data['PL'])
total_data_power = np.array(total_data['power'])
total_data_power_fraction = np.array(total_data['power fraction'])
total_data_uncertainties = np.array(total_data['sigma'])
total_data_matisse_power = np.array(total_data['Matisse Power (uW)'])

# if you want to plot the parameters only
# fit_try = objective_func((total_data_x, total_data_power_fraction, total_data_matisse_power), **kwp2)
# fit_try2 = objective_func((four_data_x, four_data_pf), **kwp2)

# popt, pcov = curve_fit(objective_func5, (total_data_x, total_data_power_fraction, total_data_matisse_power), total_data_y/total_data_power_fraction, p0=list(kwp2.values()))#, sigma=total_data_uncertainties, absolute_sigma=True)
# popt, pcov = curve_fit(objective_func6, (total_data_x, total_data_power_fraction), total_data_y/total_data_power_fraction, p0=list(kwp6.values()), sigma=total_data_uncertainties, absolute_sigma=True)

popt, pcov = curve_fit(objective_func, (total_data_x, total_data_power_fraction), total_data_y/total_data_power_fraction, p0=list(kwp2.values()),
                       bounds=([0, 0, 0, 0, 0.01, 1, 1, 0, 0, 0, 1, 2, 5, 5, 1000, 1700, 4000, 5800],
                               [20, 110, 120, 200, 20, 20, 50, 2*np.pi, np.inf, np.inf, 4, 6, 15, 15, 1700, 2500, 5000, 7000]))

# popt, pcov = curve_fit(obj2_func, (total_data_x, total_data_power_fraction), total_data_y/total_data_power_fraction, p0=list(kwp2.values()),
#                        bounds=([1E-4, 0], [1, np.pi]))

print(popt)

fit_try2 = objective_func((four_data_x, four_data_pf), *popt)

plt.yscale('log')
plt.ylim([1000, 25600])
for power in np.sort(total_data['power'].unique()):
    power_fraction = np.sqrt(power/power_reference)
    indeces = four_data_pf == power_fraction
    plt.plot(four_data_x[indeces], fit_try2[indeces]*power_fraction)

save_array = np.asarray(np.transpose([four_data_x, four_data_pf, fit_try2]))
np.savetxt('fit_data.csv', save_array)

plt.ylim([1000, 25600])
plt.yscale('log')
plt.show()

# to plot the saved data
# for power_fraction in np.sort(power_fraction_data.unique()):
#     indeces = power_fraction_data == power_fraction
#     plt.plot(x_data[indeces], fit_data[indeces]*power_fraction)
