from typing import Union

import numpy as np
import qutip as qt


def get_total_lindblad_dissipator(c_ops, data_only=False):
    """
    Uses qutip to retrieve the Lindblad dissipator.
    """
    return sum([qt.lindblad_dissipator(qt.Qobj(op), data_only=data_only) for op in c_ops])


def get_rho_matrix_elements(hilbert_space_size):
    """
    Returns an np.ndarray of shape (hss**2, hss, hss) were each subarray of shape (hss, hss) contains a matrix of all
    0 except one element that has value 1 for all hss**2 possible elements in the matrix
    (where hss == hilbert_space_size)
    """
    return np.reshape(np.diag(np.ones(hilbert_space_size ** 2, dtype=np.complex128)),
                      (hilbert_space_size ** 2, hilbert_space_size, hilbert_space_size))


def get_hamiltonian_density_matrix_commutator(hamiltonian, rho_matrix_elements: np.ndarray):
    """
    Returns the [H, rho] = (H*rho-rho*H) part of the Liouville - Von Neumann matrix elements as a (hss**2, hss*2) matrix
    It receives the rho_matrix_elements as a parameter, so that when we call the function many, times, we will save time
    from creating the matrix elements over and over again.
    """
    hss2 = rho_matrix_elements.shape[0]
    return np.array([(hamiltonian @ rho_matrix_elements[i] - rho_matrix_elements[i] @ hamiltonian).reshape(hss2)
                     for i in range(hss2)])


def get_master_equation_matrix_elements(h_rho_com, lindbladian):
    """
    Returns the matrix elements of the matrix equation by using the [H, rho] commutator and the sum of Lindblad
     dissipators.
    """
    drhodt = -1j * h_rho_com + lindbladian
    # sets the last line into population conservation
    size = drhodt.shape[0]
    hilbert_space_size = int(np.sqrt(size))
    drhodt[-1] = 0  # sets all values to 0
    drhodt[-1, [(hilbert_space_size + 1) * i for i in range(hilbert_space_size)]] = np.ones(hilbert_space_size)
    return drhodt


def get_initial_rho_as_vec_from_state(initial_state):
    """
    Using the initial state ket, it returns equivalent the initial density matrix.
    """
    initial_state = np.array(initial_state).reshape((-1, 1))  # get a 2D matrix of the initial_state
    return (initial_state @ initial_state.transpose()).reshape(-1)  # get initial rho as vector


def get_rhs_vector(hilbert_space_size):
    """
    Returns a vector full of zeros except the last element depending on the hilbert space size. This will be the right
    hand side of the master equation, which is set to 0 for steady state (except the last one that is used for
    population conservation.
    """
    vec = np.zeros(hilbert_space_size ** 2)
    vec[-1] = 1
    return vec


def get_steady_state_solution(drhodt, rhs_vec):
    """
    Provided the right hand side vector and the left hand side vector or "master equation matrix elements" (as dρ/dt),
    this function returns the steady state density matrix solution.
    """
    hilbert_space_size = int(np.sqrt(rhs_vec.shape[0]))
    return np.linalg.solve(drhodt, rhs_vec).reshape(hilbert_space_size, hilbert_space_size)


def find_solution(hamiltonians: np.ndarray, lindbladian: np.ndarray, rho_matrix_elements: np.ndarray,
                  rhs_vector: np.ndarray):
    """
    This function uses the Hamiltonian(s) and Lindbladian, finds the steady state solution(s) of the system.
    Takes either a 2-dimensional Hamiltonian as an ndarray or a assortment of Hamiltonians (a 3-dimensional np.ndarray).
    """
    if len(hamiltonians.shape) == 2:
        hamiltonian = hamiltonians
        H_rho_commutator = get_hamiltonian_density_matrix_commutator(hamiltonian, rho_matrix_elements)
        master_equation_matrix_elements = get_master_equation_matrix_elements(H_rho_commutator, lindbladian)
        return get_steady_state_solution(master_equation_matrix_elements, rhs_vector)
    elif len(hamiltonians.shape) == 3:
        H_rho_commutators = [get_hamiltonian_density_matrix_commutator(hamiltonian, rho_matrix_elements)
                             for hamiltonian in hamiltonians]
        master_equations_matrix_elements = [get_master_equation_matrix_elements(H_rho_commutator, lindbladian)
                                            for H_rho_commutator in H_rho_commutators]
        return np.array([get_steady_state_solution(master_equation_matrix_elements, rhs_vector)
                         for master_equation_matrix_elements in master_equations_matrix_elements])
    else:
        print('The hamiltonian dimensions must be either 2 or 3.')
        return np.zeros(hamiltonians.shape)


def find_solution_broadened(hamiltonians: np.ndarray, lindbladian: np.ndarray, rho_matrix_elements: np.ndarray,
                            rhs_vector: np.ndarray, probability_profile: np.ndarray):
    """
    Provided a list of hamiltonians with different broadening, the lindbladian and the probability profile of the
    broadening, this function returns the sum the differently broadened solutions, after multiplying each subsolution
    with its the probability profile. That means the return np.ndarray is one less dimension than the given hamiltonian.

    Takes either a 3-dimentional assortment of Hamiltonians (where the first dimension is the one related to the
    broadening and the next two represent each hamiltonian) or a 4-dimentional assortment of Hamiltonians (where the
    second dimension is related to the broadening and the 3rd and 4th represent each hamiltonian).
    """
    if len(hamiltonians.shape) == 3:
        sol = find_solution(hamiltonians, lindbladian, rho_matrix_elements, rhs_vector)
        return np.sum(probability_profile[:, np.newaxis, np.newaxis] * sol, axis=0)
        # return sum(probability_profile*sol)
    if len(hamiltonians.shape) == 4:
        sol = np.array([find_solution(hamiltonians_broadened, lindbladian, rho_matrix_elements, rhs_vector)
                        for hamiltonians_broadened in hamiltonians])
        return np.sum(probability_profile[np.newaxis, :, np.newaxis, np.newaxis] * sol, axis=1)
        # return sum(probability_profile*sol)


def solve_system(hamiltonians: np.ndarray, c_ops: np.ndarray, broadening_probability_profile: np.ndarray = None):
    """
    Given the Hamiltonians, the c_ops of the Lindbladian and the probability profile of the broadening (if any)
    and returns the density matrix steady state solutions for each hamiltonian.
    """
    hilbert_space_size = hamiltonians.shape[-1]
    Ltot = np.array(get_total_lindblad_dissipator(c_ops))
    rho_matrix_elements = get_rho_matrix_elements(hilbert_space_size)
    rhs_vec = get_rhs_vector(hilbert_space_size)
    if broadening_probability_profile is not None:
        if broadening_probability_profile.shape[0] == hamiltonians.shape[-3]:
            return find_solution_broadened(hamiltonians, Ltot, rho_matrix_elements, rhs_vec,
                                           broadening_probability_profile)
        else:
            print('The probability_profile size does not much the third from the end dimension size of the'
                  ' Hamiltonians.')
            return None
    else:
        return find_solution(hamiltonians, Ltot, rho_matrix_elements, rhs_vec)


def retrieve_part_of_solution(solution: np.ndarray, rows, columns):
    """
    Takes a 3-dimensional ndarray as input and outputs only the given row or column.
    E.g. if row = 2 and column = 3, this function will return "solution[:, 2, 3]"
         if  row = [2, 4] and column = [3, 1], this function will return "solution[:, 2, 3] + solution[:, 4, 1]"
    """

    def is_iterable(obj):
        try:
            iter(obj)
            return True
        except TypeError:
            return False

    if not is_iterable(rows):
        rows = [rows]

    if not is_iterable(columns):
        columns = [columns]

    if len(rows) != len(columns):
        print('Length of row and column lists is not the same.')
        return np.array([])
    else:
        solution_sum = np.zeros((solution.shape[0],))
        for i in range(len(rows)):
            if rows[i] < solution.shape[1] and columns[i] < solution.shape[2]:
                solution_sum += np.array(solution[:, rows[i], columns[i]]).real
        return solution_sum


def smart_frequency_range_choice_around_resonances(important_centers: Union[list, float],
                                                   starting_order_of_magnitude: float = -4,
                                                   ending_order_of_magnitude: float = 2,
                                                   points_per_magnitude: int = 10):
    if isinstance(important_centers, float):
        important_centers = [important_centers]

    pos = np.logspace(starting_order_of_magnitude, ending_order_of_magnitude,
                      int(np.ceil((ending_order_of_magnitude - starting_order_of_magnitude + 1) * points_per_magnitude
                                  + 1)))

    range_around_center = np.array(list(-pos[::-1]) + [-pos[0] / 2, 0, pos[0] / 2] + list(pos))
    total = np.array([])
    for center in important_centers:
        total = np.append(total, range_around_center + center)
    return np.sort(total)
