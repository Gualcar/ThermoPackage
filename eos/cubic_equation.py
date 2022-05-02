import numpy as np

from math import log

from auxiliar.root_finding import roots_n3

def cubic_equation(A, B, pressure, temperature, composition):
    """

    :param eps:
    :param sig:
    :param bet:
    :param qm:
    :param is_liquid_phase:
    :return:
    """

    # TODO - give better meaning of each attribute.
    # TODO - Is it the same for all EOS formulations. If not, needs refactoring.

    T1 = -(1 - B)
    T2 = A - 3*B**2 - 2*B
    T3 = -(A*B - B**2 - B**3)

    coefs = np.array([1.0, T1, T2, T3], dtype=np.complex128)

    _roots = roots_n3(1.0, T1, T2, T3)

    Z_roots = np.real(_roots[np.abs(np.imag(_roots)) < 1e-9])


    if len(Z_roots) == 1:
        Z = Z_roots[0]
    elif (len(Z_roots) == 2 or len(Z_roots) == 3):
        Z1 = min(abs(Z_roots))
        Z2 = max(abs(Z_roots))


        # Expressão retirada Abbas 2016

        # s0 = 0
        # for i in range(len(composition)):
        #     s1 = 0
        #     for j in range(len(composition)):
        #         s1 += composition[j]*aij[i,j]
        #     s0 += composition[i]*((bp[i]/bm)*(Z2-Z1) - log((Z2-B)/(Z1-B)) - (A/(2.82*B))*(2*s0/am - bp[i]/bm)*log(((Z2 + 2.414*B)*(Z1 - 0.414*B))/((Z2 - 0.414*B)*(Z1 + 2.414*B))))
        # delta_g = R*T*s0

        # Expressão retirada de CMG, 2016, Win Prop Manual
        #Appendix B - 185
        eps1 = 2.414
        eps2 = -0.414

        try:
            delta_g = log((Z2 - B)/(Z1 - B)) + (1/(eps2-eps1))*(A/B)*log(((Z2+eps2*B)/(Z1+eps2*B))*((Z1+eps1*B)/(Z2+eps1*B))) - (Z2 - Z1)    
        except:
            delta_g = log(abs(Z2 - B)/abs(Z1 - B)) + (1/(eps2-eps1))*(A/B)*log(abs(((Z2+eps2*B)/(Z1+eps2*B))*((Z1+eps1*B)/(Z2+eps1*B)))) - (Z2 - Z1) 


        if delta_g > 0:
            Z = Z2
        else:
            Z = Z1

    return Z

# from numba import vectorize, float32, float64, boolean
# @vectorize([float32(float32, float32, float32, float32, boolean),
#             float64(float64, float64, float64, float64, boolean)])
# def cubic_equation_vectorized(eps, sig, bet, qm, is_liquid_phase):
#
#     A = (eps + sig) * bet - 1.0 - bet
#     B = eps * sig * bet ** 2 - (eps + sig) * (bet + 1) * bet + qm * bet
#     C = -qm * bet ** 2 - eps * sig * (bet ** 2) * (bet + 1)
#
#     coefs = np.array([1.0, A, B, C], dtype=np.complex128)
#
#     Z_roots = np.roots(coefs)
#
#     Z_roots = [np.real(z) for z in Z_roots if np.imag(z) != 0.0]
#
#     if len(Z_roots) == 1 and Z_roots[0] > 0.5 and not is_liquid_phase:
#         Z = Z_roots[0]
#     elif len(Z_roots) == 1 and Z_roots[0] <= 0.5 and is_liquid_phase:
#         Z = Z_roots[0]
#     elif is_liquid_phase:
#         Z = min(Z_roots)
#     else:
#         Z = max(Z_roots)
#
#     return Z