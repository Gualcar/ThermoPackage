import numpy as np

from scipy import optimize
from math import exp

#Methane density based on Extended BWR EoS
#See ali et al 1991

def BWR_equation(rho, *args):
    temperature, pressure, a = args

    sum1 = np.zeros_like(temperature)
    sum2 = np.zeros_like(temperature)

    for k in range(len(temperature)):
        for i in range(9):
            sum1[k] += a[i,k]*rho[k]**(i+1)
        for i in range(9,15):
            sum2[k] += a[i,k]*(rho[k]**(2*(i+1)-17))*exp(-0.0096*rho[k]**2)

    res = pressure - sum1 - sum2

    return res

def methane_density(temperature, pressure):

    #Parameters
    R = 0.08205616 # atm/molK

    #EoS parameters
    a = np.empty([15,len(temperature)])
    N = np.empty(32)

    N[0] = -1.8439486666e-2
    N[1] = 1.0510162064 
    N[2] = -16.057820303
    N[3] = 8.4844027562e2
    N[4] = -4.2738409106e4
    N[5] = 7.6565285254e-4
    N[6] = -4.8360724197e-1
    N[7] = 85.195473835
    N[8] = -1.6607434721e4
    N[9] = -3.7521074532e-5
    N[10] = 2.8616309259e-2
    N[11] = -2.685285973
    N[12] = 1.1906973942e-4
    N[13] = -8.5315715699e-3
    N[14] = 3.8365063841
    N[15] = 2.4986828379e-5
    N[16] = 5.7974531455e-6
    N[17] = -7.16483292927e-3
    N[18] = 1.2577853784e-4
    N[19] = 2.2240102466e4
    N[20] = -1.800512328e6
    N[21] = 50.498054887
    N[22] = 1.6428375992e6
    N[23] = 2.1325387196e-1
    N[24] = 37.791273422
    N[25] = -1.1857016815e-5
    N[26] = -36.30780767
    N[27] = -4.1006782941e-6
    N[28] = 1.4870043284e-3
    N[29] = 3.1512261532e-9
    N[30] = -2.1670774745e-6
    N[31] = 2.400055179e-5

    a[0,:] = R*temperature
    a[1,:] = N[0]*temperature + N[1]*temperature**0.5 + N[2] + N[3]*temperature + N[4]/temperature**2
    a[2,:] = N[5]*temperature + N[6] + N[7]/temperature + N[8]/temperature**2
    a[3,:] = N[9]*temperature + N[10] + N[11]/temperature
    a[4,:] = N[12]
    a[5,:] = N[13]/temperature + N[14]/temperature**2
    a[6,:] = N[15]/temperature
    a[7,:] = N[16]/temperature + N[17]/temperature**2
    a[8,:] = N[18]/temperature**2
    a[9,:] = N[19]/temperature**2 + N[20]/temperature**3
    a[10,:] = N[21]/temperature**2 + N[22]/temperature**4
    a[11,:] = N[23]/temperature**2 + N[24]/temperature**3
    a[12,:] = N[25]/temperature**2 + N[26]/temperature**4
    a[13,:] = N[27]/temperature**2 + N[28]/temperature**3
    a[14,:] = N[29]/temperature**2 + N[30]/temperature**3 + N[31]/temperature**4

    #Guess the density
    x0 = np.ones_like(temperature)

    #Solve BWR equation to find the root
    resp = optimize.root(BWR_equation, x0, args=(temperature, pressure, a))

    return resp.x


if __name__ == '__main__':

    temperature = np.array([293.0,293.0])
    pressure = np.array([1.0,1.0])

    print(methane_density(temperature, pressure))