import numpy as np
from matplotlib import pyplot as plt

from steadyqusim import steady_state_quantum_system_solver as ssqss
from steadyqusim.hamiltonians.ZnO_shallow_donor_hamiltonians import hamiltonian_ZnO_Voigt_neglect14 as \
    hamiltonian
from example_ZnO_CPT_Voigt_xy_constants_and_parameters import Del13, Del23, Om13, Om23, Del24, Om24, c_ops
from steadyqusim.broadening_functions import get_ihhomogeneous_broadening_parameters

hamiltonians = np.array([hamiltonian(delta13=Del13, delta23=d23, omega13=Om13, omega23=Om23, delta24=d24, omega24=Om24)
                         for (d23, d24) in zip(Del23, Del24)])

solution = ssqss.solve_system(hamiltonians, c_ops)
excited_states_population = ssqss.retrieve_part_of_solution(solution, [2, 3], [2, 3])


Delta_broadenings, prb_profiles, _ = get_ihhomogeneous_broadening_parameters(4, len(Del23))

broadened_hamiltonians = np.array([[hamiltonian(delta13=Del13, delta23=d23, omega13=Om13, omega23=Om23, delta24=d24,
                                                omega24=Om24, delta_broadening=db)
                                    for db in Delta_broadenings]
                                   for (d23, d24) in zip(Del23, Del24)])

broadened_solution = ssqss.solve_system(broadened_hamiltonians, c_ops, prb_profiles)

broadened_excited_states_population = ssqss.retrieve_part_of_solution(broadened_solution, [2, 3], [2, 3])

plt.plot(Del23/2/np.pi, excited_states_population)
plt.plot(Del23/2/np.pi, broadened_excited_states_population)
plt.show()
