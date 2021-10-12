import numpy as np


# def Voigt_neglect14(d23, dbr=0., om23=Om23, om13=Om13, d24=Del24, om24=Om24):
#
#     return np.array([[d23 - Del13, 0, -om13 / 2, 0],
#                      [0, 0, -om23 / 2, -om24 / 2],
#                      [-om13 / 2, -om23 / 2, d23 - dbr, 0],
#                      [0, -om24 / 2, 0, d24 - dbr]])
#

def hamiltonian_ZnO_Voigt_neglect14(delta13, delta23, omega13, omega23, delta_excited=0, delta_ground=0,
                                    delta_broadening=0, delta24=None, omega24=None):
    """
    The ZnO shallow donor Hamiltonian in a magnetic field in Voigt geometry has 4 transisitions of two different
    polarizaitons. That means that with two lasers on, there are 4 different transitions with non-zero elements, not
    permitting a steady state solution. In this hamiltonian, we neglect the the 1-4 transition.
    """

    if delta24 is None:
        delta24 = delta_excited - delta_ground + delta13
    if omega24 is None:
        omega24 = omega13

    return np.array([[delta23 - delta13, 0, -omega13 / 2, 0],
                     [0, 0, -omega23 / 2, -omega24 / 2],
                     [-omega13 / 2, -omega23 / 2, delta23 - delta_broadening, 0],
                     [0, -omega24 / 2, 0, delta24 - delta_broadening]])


def hamiltonian_ZnO_Voigt_neglect24(delta13, delta23, omega13, omega23, delta_excited, delta_ground,
                                    delta_broadening=0, delta14=None, omega14=None):
    """
    The ZnO shallow donor Hamiltonian in a magnetic field in Voigt geometry has 4 transisitions of two different
    polarizaitons. That means that with two lasers on, there are 4 different transitions with non-zero elements, not
    permitting a steady state solution. In this hamiltonian, we neglect the the 2-4 transition.
    """

    if delta14 is None:
        delta14 = delta_excited + delta_ground + delta23
    if omega14 is None:
        omega14 = omega23

    return np.array([[0, 0, -omega13 / 2, -omega14 / 2],
                     [0, delta13 - delta23, -omega23 / 2, 0],
                     [-omega13 / 2, -omega23 / 2, delta13 - delta_broadening, 0],
                     [-omega14 / 2, 0, 0, delta14 - delta_broadening]])


def hamiltonian_three_level(delta1, delta2, omega1, omega2, delta_broadening=0):
    """
    An effective three-level Hamiltonian for a degenerate excited state like indium in ZnO nanowires
    """

    return np.array([[0, 0, omega1/2],
                     [0, delta1-delta2, omega2/2],
                     [omega1/2, omega2/2, delta1-delta_broadening]])


def hamiltonian_four_level_degenerate_excited_state(delta1, delta2, omega1, omega2, phi, delta_broadening=0):
    """
    An effective three-level Hamiltonian for a degenerate excited state like indium in ZnO nanowires
    """

    return np.array([[0, 0, omega1/2, omega1/2],
                     [0, delta1-delta2, omega2/2, omega2/2*np.exp(-1j*phi)],
                     [omega1/2, omega2/2, delta1-delta_broadening, 0],
                     [omega1/2, omega2/2*np.exp(1j*phi), 0, delta1-delta_broadening]])

