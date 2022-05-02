from re import S
import numpy as np

from scipy import optimize

from eos.cubic_equation import cubic_equation

def calculate_viscosity_simple(rho:np.ndarray): 
    if np.all(rho > 150):
        return np.ones_like(rho)*1.26e-3
    else:
        return np.ones_like(rho)*1.195e-5


def calculate_viscosity(composition:np.ndarray, temperature: np.ndarray, pressure: np.ndarray, pure_molar_density: np.ndarray, molar_density: np.ndarray, props): 

    # Viscosity correlation based on Chung et al 1988
    # The correlation proposed here is only suitable for non-associative mixtures

    #Reshaping arrays
    temperature = temperature.reshape([-1,1])
    pressure = pressure.reshape([-1,1])
    molar_density = molar_density.reshape([-1,1])

    pure_molar_density = pure_molar_density.T 
    #===============================
    #========  Dilute Gas ==========
    #===============================

    #Boltzman constant
    k = 1.38064e-23

    #Potencial distance
    sigma = 0.809*props.Vc**(1/3)
    #Potencial energy parameter
    epsilon = k*props.Tc/1.2593

    temperature_star = k*temperature/epsilon

    #Calculate the collision integral 
    A = 1.16145
    B = 0.14874
    C = 0.52487
    D = 0.77320
    E = 2.16178
    F = 2.43787
    G = -6.435e-4
    H = 7.27371
    S = 18.0323 
    W = -0.76830
    omega_star = A/(temperature_star**B) + \
                 C/np.exp(D*temperature_star)  + \
                 E/np.exp(F*temperature_star) + \
                 G*(temperature_star**B)*np.sin(S*temperature_star**W - H)

    #Consultar artigo de Reid 1977 para pegar os termos ni

    correction_factor = 1 - 0.2756*props.w

    dilute_gas_viscosity = 4.0785e-5*np.sqrt(props.MM*temperature)*correction_factor/(omega_star*props.Vc**(2/3))

    #Consultar artigos Fakeeha 1983, Passut 1972, Lee 1981 para cálculo de cv de gás ideal
    psi = 1 # Todo: cálculo de psi necessita de Cv de gás ideal
    dilute_gas_thermal_conductivity = 7.452*(dilute_gas_viscosity/props.MM)*psi

    #=================================
    #========  Dense Fluids ==========
    #=================================

    #Parameters a
    a = np.empty([10,2])
    a[0,:] = np.array([6.32402, 50.4119])
    a[1,:] = np.array([0.12102e-2, -0.11536e-2])
    a[2,:] = np.array([5.28346, 254.209])
    a[3,:] = np.array([6.62263, 38.0957])
    a[4,:] = np.array([19.7454, 7.63034])
    a[5,:] = np.array([-1.89992, -12.5367])
    a[6,:] = np.array([24.2745, 3.44945])
    a[7,:] = np.array([0.79716, 1.11764])
    a[8,:] = np.array([-0.23816, 0.067695])
    a[9,:] = np.array([0.068629, 0.34793])

    pure_A = np.empty([10,len(props.w)])
    for i in range(len(props.w)):
        pure_A[:,i] = a[:,0] + a[:,1]*props.w[i]
    

    Y = 1e-6*pure_molar_density*props.Vc/6
    G1 = (1 - 0.5*Y)/(1 - Y)**3
    G2 = (pure_A[0,:]*(1 - np.exp(-pure_A[3,:]*Y)) + pure_A[1,:]*G1*np.exp(pure_A[4,:]*Y) + pure_A[2,:]*G1)/(pure_A[0,:]*pure_A[3,:] + pure_A[1,:] + pure_A[2,:])

    eta_k = dilute_gas_viscosity*(1/G2 + pure_A[5,:]*Y)
    eta_p = (36.344e-6*np.sqrt(props.MM*props.Tc)/props.Vc**(2/3))*pure_A[6,:]*(Y**2)*G2*np.exp(pure_A[7,:] + pure_A[8,:]/temperature_star + pure_A[9,:]/temperature_star**2)

    dense_fluid_viscosity = eta_k + eta_p

    #=======================================
    #========  Mixture properties ==========
    #=======================================

    binary_sigma = np.empty([len(props.MM),len(props.MM)])
    binary_epsilon = np.empty_like(binary_sigma)
    binary_accentric = np.empty_like(binary_sigma)
    binary_molar_mass = np.empty_like(binary_sigma)

    for i in range(len(props.MM)):
        for j in range(len(props.MM)):
            binary_sigma[i,j] = (sigma[i]*sigma[j])**0.5
            binary_epsilon[i,j] = k*(epsilon[i]*epsilon[j])*0.5
            binary_accentric[i,j] = 0.5*(props.w[i] + props.w[j])
            binary_molar_mass[i,j] = 2*props.MM[i]*props.MM[j]/(props.MM[i] + props.MM[j])

    mixture_sigma = np.zeros(len(temperature))

    for t in range(len(temperature)):
        for i in range(len(props.MM)):
            for j in range(len(props.MM)):
                mixture_sigma[t] += composition[i,t]*composition[j,t]*binary_sigma[i,j]**3
    
    mixture_epsilon = np.zeros(len(temperature))

    for t in range(len(temperature)):
        for i in range(len(props.MM)):
            for j in range(len(props.MM)):
                mixture_epsilon[t] += composition[i,t]*composition[j,t]*binary_epsilon[i,j]*(binary_sigma[i,j]**3)/(mixture_sigma[t]**3)
    
    mixture_accentric = np.zeros(len(temperature))

    for t in range(len(temperature)):
        for i in range(len(props.MM)):
            for j in range(len(props.MM)):
                mixture_accentric[t] += composition[i,t]*composition[j,t]*binary_accentric[i,j]*(binary_sigma[i,j]**3)/(mixture_sigma[t]**3)

    mixture_MM = np.zeros(len(temperature))

    for t in range(len(temperature)):
        for i in range(len(props.MM)):
            for j in range(len(props.MM)):
                mixture_MM[t] += (composition[i,t]*composition[j,t]*binary_epsilon[i,j]*(binary_sigma[i,j]**2)*(binary_molar_mass[i,j]**0.5)/(mixture_epsilon[t]*mixture_sigma[t]**2))**2

    mixture_Vc = (mixture_sigma/0.809)**3
    mixture_Tc = (1.2593*mixture_epsilon)/k

    #=============================================
    #Calculate the same parameters for the mixture
    temperature = temperature.flatten()
    molar_density = molar_density.flatten() 

    mixture_temperature_star = k*temperature/mixture_epsilon

    mixture_omega_star = A/(mixture_temperature_star**B) + \
                 C/np.exp(D*mixture_temperature_star)  + \
                 E/np.exp(F*mixture_temperature_star) + \
                 G*(mixture_temperature_star**B)*np.sin(S*mixture_temperature_star**W - H)

    #Consultar artigo de Reid 1977 para pegar os termos ni

    correction_factor = 1 - 0.2756*mixture_accentric

    mixture_dilute_gas_viscosity = 4.0785e-5*np.sqrt(mixture_accentric*temperature)*correction_factor/(mixture_omega_star*mixture_Vc**(2/3))

    mixture_A = np.empty([10,len(temperature)])
    for i in range(len(temperature)):
        mixture_A[:,i] = a[:,0] + a[:,1]*mixture_accentric[i]
    
    mixture_Y = 1e-6*molar_density*mixture_Vc/6
    mixture_G1 = (1 - 0.5*mixture_Y)/(1 - mixture_Y)**3
    mixture_G2 = (mixture_A[0,:]*(1 - np.exp(-mixture_A[3,:]*mixture_Y)) + mixture_A[1,:]*mixture_G1*np.exp(mixture_A[4,:]*mixture_Y) + mixture_A[2,:]*mixture_G1)/(mixture_A[0,:]*mixture_A[3,:] + mixture_A[1,:] + mixture_A[2,:])

    eta_k = mixture_dilute_gas_viscosity*(1/mixture_G2 + mixture_A[5,:]*mixture_Y)
    eta_p = (36.344e-6*np.sqrt(mixture_MM*mixture_Tc)/mixture_Vc**(2/3))*mixture_A[6,:]*(mixture_Y**2)*mixture_G2*np.exp(mixture_A[7,:] + mixture_A[8,:]/mixture_temperature_star + mixture_A[9,:]/mixture_temperature_star**2)

    mixture_dense_fluid_viscosity = eta_k + eta_p

    return mixture_dense_fluid_viscosity


def calculate_viscosity_ali(composition:np.ndarray, temperature: np.ndarray, pressure: np.ndarray, props):

    temperature = np.array([temperature])
    pressure = np.array([pressure])

    #Methane Critical Properties
    methane_Pc = 45.99
    methane_Tc = 190.6
    methane_MM = 16.043
    critical_methane = 162.8e-3

    #Correlation based on Ali 1991

    #Fluid correlation will be methane
    #Instances to evaluate the density of methane

    #Appendix B - Pedersen correlation
    #Mixture parameters
    numerator = np.zeros_like(temperature)
    denominator = np.zeros_like(temperature)
    MWn = np.zeros_like(temperature)
    MWw_numerator = np.zeros_like(temperature)
    MWw = np.zeros_like(temperature)
    for k in range(len(temperature)):
        for i in range(len(props.Pc)):
            MWn[k] += composition[i,k]*props.MM[i]
            MWw_numerator[k] += composition[i,k]*(props.MM[i]**2)
            for j in range(len(props.Pc)):
                denominator[k] += composition[i,k]*composition[j,k]*((props.Tc[i]/props.Pc[i])**(1/3)+(props.Tc[j]/props.Pc[j])**(1/3))**3
                numerator[k] += composition[i,k]*composition[j,k]*(((props.Tc[i]/props.Pc[i])**(1/3)+(props.Tc[j]/props.Pc[j])**(1/3))**3)*((props.Tc[i]*props.Tc[j])**0.5)           

    Tcmix = numerator/denominator 
    Pcmix = 8*numerator/(denominator**2)
    MWw = MWw_numerator/MWn
    MWmix = 1.304e-4*(MWw**2.303 - MWn**2.303) + MWn

    temperature_rho = temperature*methane_Tc/Tcmix
    pressure_rho = pressure*methane_Pc/Pcmix

    #Evaluate density for the temperature and pressure above
    #critical_methane = BWR_methane(np.ones_like(temperature)*methane_Tc, np.ones_like(temperature)*methane_Pc)
    # reference_density = BWR_methane(temperature_rho, pressure_rho)
    # pure_methane = BWR_methane(temperature, pressure)
    reference_density = peng_robinson_methane(temperature_rho, pressure_rho)
    pure_methane = peng_robinson_methane(temperature, pressure)


    #Transform into g/m³
    reference_density = reference_density*1e-3 
    pure_methane = pure_methane*1e-3
    #critical_methane = critical_methane*1e-3 

    reduced_density = reference_density/critical_methane

    alfa0 = 1 + 8.374e-4*reduced_density**4.5265
    alfamix = 1 + 7.378e-3*(reduced_density**1.847)*(MWmix**0.5173)

    equivalent_pressure = pressure*methane_Pc*alfa0/(Pcmix*alfamix)
    equivalent_temperature = temperature*methane_Tc*alfa0/(Tcmix*alfamix)

    #Ali also proposed this expression with different parameters

    # Hanley et al 1975 with correction of Ali 1991
    # Calculate reference viscosity
    reference_viscosity = Hanley_methane(equivalent_temperature, equivalent_pressure)

    nimix = reference_viscosity*((Tcmix/methane_Tc)**(-1/6))*((Pcmix/methane_Pc)**(2/3))*((MWmix/methane_MM)**(2/3))*(alfamix/alfa0)

    return nimix*1e-6


#Appendix D - Extended BWR-equation for the reference fluid methane

def BWR_methane(temperature, pressure):
    
    #Change pressure: bar -> atm
    pressure = pressure*(1/1.01325) 

    #BWR parameters
    R = 0.08205616
    gamma = 0.0096

    #Check this parameters - Difference between Ali 1991 and Ely 1981
    N = np.array([
         -1.8439486666e-2,
          1.0510162064,
          -1.6057820303e1,
          8.4844027562e2,
          -4.2738409106e4,
          7.6565285254e-4,
          -4.8360724197e-1,
          8.5195473835e1,
          -1.6607434721e4,
          -3.7521074532e-5,
          2.8616309259e-2,
          -2.8685285973, 
          1.1906973942e-4,
          -8.5315715699e-3,
          3.8365063841, 
          2.4986828379e-5,
          5.7974531455e-6,
          -7.6483292927e-3,
          1.2577853784e-4,
          2.2240102466e4,
          -1.4800512328e6,
          5.0498054887e1,
          1.6428375992e6,
          2.1325387196e-1,
          3.7791273422e1,
          -1.1857016815e-5,
          -3.1630780767e1,
          -4.1006782941e-6,
          1.4870043284e-3,
          3.1512261532e-9,
          -2.1670774745e-6,
          2.400055179e-5
    ])

    def residue(z_factor):

        density = pressure/(z_factor*R*temperature)

        equation = -z_factor*R*temperature*density + \
                density*R*temperature + \
                (density**2)*(N[0]*temperature + N[1]*temperature**0.5 + N[2] + N[3]/temperature + N[4]/(temperature**2)) + \
                (density**3)*(N[5]*temperature + N[6] + N[7]/temperature + N[8]/(temperature**2)) + \
                (density**4)*(N[9]*temperature + N[10] + N[11]/temperature) + \
                (density**5)*(N[12]) + \
                (density**6)*(N[13]/temperature + N[14]/(temperature**2)) + \
                (density**7)*(N[15]/temperature) + \
                (density**8)*(N[16]/temperature + N[17]/(temperature**2)) + \
                (density**9)*(N[18]/(temperature**2)) + \
                (density**3)*(N[19]/(temperature**2) + N[20]/(temperature**3))*np.exp(-gamma*(density**2)) + \
                (density**5)*(N[21]/(temperature**2) + N[22]/(temperature**4))*np.exp(-gamma*(density**2)) + \
                (density**7)*(N[23]/(temperature**2) + N[24]/(temperature**3))*np.exp(-gamma*(density**2)) + \
                (density**9)*(N[25]/(temperature**2) + N[26]/(temperature**4))*np.exp(-gamma*(density**2)) + \
                (density**11)*(N[27]/(temperature**2) + N[28]/(temperature**3))*np.exp(-gamma*(density**2)) + \
                (density**13)*(N[29]/(temperature**2) + N[30]/(temperature**3) + N[31]/(temperature**4))*np.exp(-gamma*(density**2))
        return equation
    
    def test_eq(density):
        pressure = density*R*temperature + \
                (density**2)*(N[0]*temperature + N[1]*temperature**0.5 + N[2] + N[3]/temperature + N[4]/(temperature**2)) + \
                (density**3)*(N[5]*temperature + N[6] + N[7]/temperature + N[8]/(temperature**2)) + \
                (density**4)*(N[9]*temperature + N[10] + N[11]/temperature) + \
                (density**5)*(N[12]) + \
                (density**6)*(N[13]/temperature + N[14]/(temperature**2)) + \
                (density**7)*(N[15]/temperature) + \
                (density**8)*(N[16]/temperature + N[17]/(temperature**2)) + \
                (density**9)*(N[18]/(temperature**2)) + \
                (density**3)*(N[19]/(temperature**2) + N[20]/(temperature**3))*np.exp(-gamma*(density**2)) + \
                (density**5)*(N[21]/(temperature**2) + N[22]/(temperature**4))*np.exp(-gamma*(density**2)) + \
                (density**7)*(N[23]/(temperature**2) + N[24]/(temperature**3))*np.exp(-gamma*(density**2)) + \
                (density**9)*(N[25]/(temperature**2) + N[26]/(temperature**4))*np.exp(-gamma*(density**2)) + \
                (density**11)*(N[27]/(temperature**2) + N[28]/(temperature**3))*np.exp(-gamma*(density**2)) + \
                (density**13)*(N[29]/(temperature**2) + N[30]/(temperature**3) + N[31]/(temperature**4))*np.exp(-gamma*(density**2))
        return pressure

    resp = optimize.root(residue, np.ones_like(temperature)*1) 
    if resp.success == False:
        resp = optimize.root(residue, np.ones_like(temperature)*50)

    density = resp.x

    #Convert to kg/m³
    density = 16.04*density

    return density


def Hanley_methane(temperature, pressure):

    #methane_density = BWR_methane(temperature, pressure)*1e-3
    methane_density = peng_robinson_methane(temperature, pressure)*1e-3

    critical_methane = 162.8e-3

    teta = (methane_density/critical_methane - 1)

    r1 = -2.090975e5     
    r2 = 2.647269e5 
    r3 = -1.472818e5
    r4 = 4.716740e4
    r5 = -9.491872e3
    r6 = 1.219979e3 
    r7 = -9.627993e1
    r8 = 4.274152
    r9 = -8.141531e-2 
    ni0_1 = r1/temperature + r2*(temperature**(-2/3)) + r3*(temperature**(-1/3)) + r4 + r5*(temperature**(1/3)) + r6*(temperature**(2/3)) + r7*temperature + r8*(temperature**(4/3)) + r9*(temperature**(5/3)) 

    s1 = 1.6969859271
    s2 = -1.3337234608e-1
    s3 = 1.4
    s4 = 168
    ni0_2 = s1 + s2*(s3 - np.log(temperature/s4))**2

    t1 = -1.035060586e1 
    t2 = 1.7571599671e1
    t3 = -3.0193918656e3
    t4 = 1.8873011594e2
    t5 = 4.2903609488e-2
    t6 = 1.4529023444e2
    t7 = 6.1276818706e3
    #Todo: avaliar essa expressão
    delta_ni0_1 = np.exp(t1 + t4/temperature)*(np.exp((t2 + t3/(temperature**(3/2)))*(methane_density**0.1) + teta*(methane_density**0.5)*(t5 + t6/temperature + t7/(temperature**2))) - 1)

    #Additional term proposed by Ali 1991, Pedersen and Fredenslund 1987
    #Evaluate the k7 therm
    k1 = -9.74602
    k2 = 18.0834
    k3 = -4126.66
    k4 = 44.6055
    k5 = 0.976544
    k6 = 81.8134
    k7 = 15649.9
    delta_ni0_2 = np.exp(k1 + k4/temperature)*(np.exp((k2 + k3/(temperature**(2/3)))*(methane_density**0.1) + teta*(methane_density**0.5)*(k5 + k6/temperature + k7/(temperature**2))) - 1)

    #Activation therms
    freezing_point = 90.4
    delta_t = temperature - freezing_point
    HTAN = (np.exp(delta_t) - np.exp(-delta_t))/(np.exp(delta_t) + np.exp(-delta_t))
    F1 = (HTAN + 1)/2
    F2 = (1 - HTAN)/2

    ni0 = ni0_1 + ni0_2*methane_density + F1*delta_ni0_1 + F2*delta_ni0_2

    return ni0


def peng_robinson_methane(temperature, pressure):
    
    #Calculate methane properties with 
    omg = 0.07780
    psi = 0.45724
    R = 8.314*1e-5
    #Methane Critical Properties
    Pc = 45.99
    Tc = 190.6
    MM = 16.043
    w = 0.012
    critical_density = 162.8e-3

    Tr = temperature/Tc

    k = 0.37464 + 1.54226 * w - 0.26992 * w ** 2

    alfa = (1 + k * (1 - Tr ** (0.5))) ** 2
    alfa_dt = -k * (1 + k * (1 - Tr ** (0.5))) / (np.sqrt(temperature*Tc))

    a = ((psi * (R * Tc) ** 2) / Pc) * alfa
    b = (omg*8.314*1e-5*Tc)/Pc

    A = a * pressure / (R * temperature) ** 2
    B = b * pressure / (R * temperature)

    Z = np.empty_like(temperature)
    for i, (Ai, Bi,) in enumerate(zip(A, B,)):
        Z_i = cubic_equation(Ai, Bi, pressure, temperature, 0)
        Z[i] = Z_i

    molrho = pressure / (Z * R * temperature)
    rho = molrho * MM * 1e-3

    return rho


if __name__ == "__main__":

    temperature = np.array([298.15,298.15])
    pressure = np.array([1,1])

