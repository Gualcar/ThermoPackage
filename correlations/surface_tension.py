import numpy as np

#Correlation for pure component based on Brock e Bird 1955
# For mixtures a mix rule is applied

def calculate_surface_tension(composition:np.ndarray, temperature: np.ndarray, pressure: np.ndarray, props):
    
    #Calculate the pure surface tension
    Tbr = props.Tn/props.Tc
    Q = 0.1196*(1 + Tbr*np.log(props.Pc/1.01325)/(1-Tbr)) - 0.279

    Tr = np.empty_like(composition)
    sigma = np.empty_like(composition)
    for i in range(len(temperature)):
        Tr[:,i] = temperature[i]/props.Tc
        sigma[:,i] = (props.Pc**(2/3))*(props.Tc**(1/3))*Q*(1-Tr[:,i])**(11/9)

    #Mixture surface tension
    sigma_mixture_test = composition*sigma
    sigma_mixture_test = sigma_mixture_test[~np.isnan(sigma_mixture_test).any(axis=1), :]
    sigma_mixture = np.sum(sigma_mixture_test, axis=0)

    return sigma_mixture
