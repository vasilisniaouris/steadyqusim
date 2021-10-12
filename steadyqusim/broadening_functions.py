import numpy as np


def gauss(x, mu, sigma):
    """
    Given x, mu and sigma, this function returns the normalized gausian distribution value at x.
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / sigma / np.sqrt(2 * np.pi)


def get_ihhomogeneous_broadening_parameters(fwhm, no_of_points):
    """
    Given the full width half maximum (FWHM) of your inhomogenous broadening and number of points, this function returns
    the frequencies of the broadening, the probability profile of each frequency and the constant frequency step that
    was used.
    """
    db_fwhm = fwhm * 2 * np.pi
    db_sig = db_fwhm / (2 * np.sqrt(2 * np.log(2)))
    db_x = np.linspace(-5 * db_sig, 5 * db_sig, no_of_points)
    db_dx = db_x[1] - db_x[0]
    db_y = gauss(db_x, 0, db_sig)
    prb_prof = db_y*db_dx

    return db_x, prb_prof, db_dx

