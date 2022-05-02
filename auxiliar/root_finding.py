import numpy as np

ROOTS_N3_U_AUX = 0.5j * np.sqrt(3)

def power_one_third(input_value):
    """
    Calculates the X**1/3 but correctly dealing with X < 0
    More information here: https://stackoverflow.com/questions/31231115/raise-to-1-3-gives-complex-number
    :param input_value:
    :return:
    """
    # Initiating output
    output_value = np.empty_like(input_value, dtype=complex)

    # Searching for real values
    ix = np.isreal(input_value)

    # Applying calculation for real values
    output_value[ix] = np.cbrt(np.real(input_value[ix]))

    # Searching for complex values
    iy = ix == False

    # Applying calculation for complex values
    output_value[iy] = input_value[iy] ** (1 / 3)

    return output_value

def round_complex(value, digits=9):
    """
    Rounding a complex number
    More information here: https://stackoverflow.com/questions/25820322/how-to-round-up-a-complex-number
    :param value:
    :param digits:
    :return:
    """
    return np.round(np.real(value), digits) + np.round(np.imag(value), digits) * 1j

def roots_n3(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, digits: int = 9):
    """
    Applies Cardano's formula to solve the cubic equation.
    Calculates the three roots of a*x**3 + b*x**2 + c*x**1 + d*x**0 = 0
    :param a: numpy array with any dimension containing the
    :param b: numpy array with any dimension
    :param c: numpy array with any dimension
    :param d: numpy array with any dimension
    :param digits: integer with the number of digits
    :ret    urn: tuple with 3 elements containing each root array
    """

    # Force complex number to guarantee a fixed format
    a, b, c, d = a+0j, b+0j, c+0j, d+0j

    V = - b / (3*a)
    Q = (3*a*c - b**2)/ (9*a**2)
    R = (9*a*b*c - 27*a**2*d - 2*b**3) / (54 * a**3)
    sqrt_D = np.sqrt(Q**3 + R**2)
    S = power_one_third(R + sqrt_D)
    T = power_one_third(R - sqrt_D)
    U = ROOTS_N3_U_AUX * (S - T)
    W = V - (S + T) / 2

    # Calculating roots
    root_0 = round_complex(V + S + T, digits)
    root_1 = round_complex(W + U, digits)
    root_2 = round_complex(W - U, digits)

    return np.array([root_0, root_1, root_2])

