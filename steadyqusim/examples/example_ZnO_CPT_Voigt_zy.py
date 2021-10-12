import numpy as np
from matplotlib import pyplot as plt

from compounted_library import steady_state_quantum_system_solver as ssqss
from compounted_library.hamiltonians.ZnO_shallow_donor_hamiltonians import hamiltonian_ZnO_Voigt_neglect24 as \
    hamiltonian
from example_ZnO_CPT_Voigt_zy_constants_and_parameters import Del13, Del23, Om13, Om23, Del14, Om14, Delta_exc, Delta_gr, c_ops
from compounted_library.broadening_functions import get_ihhomogeneous_broadening_parameters

hamiltonians = np.array([hamiltonian(delta13=Del13, delta23=d23, omega13=Om13, omega23=Om23, delta_excited=Delta_exc,
                                     delta_ground=Delta_gr, delta_broadening=0, delta14=Del14, omega14=Om14)
                         for d23 in Del23])
# print(Del14)
solution = ssqss.solve_system(hamiltonians, c_ops)
excited_states_population = ssqss.retrieve_part_of_solution(solution, [2, 3], [2, 3])


# Delta_broadenings, prb_profiles, _ = get_ihhomogeneous_broadening_parameters(4, len(Del23))

# broadened_hamiltonians = np.array([[hamiltonian(delta13=Del13, delta23=d23, omega13=Om13, omega23=Om23, delta_excited=Delta_exc,
#                                      delta_ground=Delta_gr, delta_broadening=db, delta14=Del14, omega14=Om14)
#                                     for db in Delta_broadenings]
#                                    for d23 in Del23])
#
# broadened_solution = ssqss.solve_system(broadened_hamiltonians, c_ops, prb_profiles)
#
# broadened_excited_states_population = ssqss.retrieve_part_of_solution(broadened_solution, [2, 3], [2, 3])
#
plt.plot(Del23/2/np.pi, excited_states_population)
# plt.plot(Del23/2/np.pi, broadened_excited_states_population)
plt.show()
