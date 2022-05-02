import numpy as np
from scipy import optimize

from auxiliar.stream import Stream
from auxiliar.phases import PhaseName
from .flash2 import Flash2


class Flash2ViaSS(Flash2):

    @staticmethod
    def flash_equation(ni, *args):
        """
        Flash function for root solver
        :param ni:
        :param args:
        :return:
        """
        z, K = args
        return np.sum(z * (K - 1) / (1 + ni * (K - 1) + 1e-12))


    def solve(self, K0):
        """

        :param composition: global composition
        :param temperature: temperature in K
        :param pressure: pressure in bar
        :param K0:
        :return:
        """

        stream = self.stream
        composition = stream.composition
        pressure = stream.P
        temperature = stream.T
        total_molar_flowrate = stream.total_molar_flowrate

        # Initializing
        diff = np.ones_like(temperature)
        cont = 0
        K = K0
        N_points = temperature.size

        # Contracting K and z shape
        K_flatten = K.reshape(K.shape[0], N_points)
        z_flatten = composition.reshape(composition.shape[0], N_points)
        total_molar_flowrate_flatten = total_molar_flowrate.reshape(N_points,1)

        # Initializing fraction and composition array
        v = np.empty_like(temperature)
        x_flatten = np.empty_like(composition)
        y_flatten = np.empty_like(composition)
        flowrate_L_flatten = np.empty_like(composition)
        flowrate_V_flatten = np.empty_like(composition)


        # Specifying and runing EOS for each phase
        flowrate_foo = np.ones_like(composition)
        self.stream_liq.specify(flowrate_foo, temperature, pressure)
        self.stream_vap.specify(flowrate_foo, temperature, pressure)


        while np.any(diff > self.tol) and cont < self.contmax:

            # TODO - It must be improved for better performance (numba candidate)

            for i in range(N_points):

                # Slicing array
                zi = z_flatten[:,i]
                Ki = K_flatten[:,i]

                # Calculating bounds
                # TODO - evaluate and compare with correspondent paper
                upper_bound = 1 / (1 - np.max(Ki))
                lower_bound = 1 / (1 - np.min(Ki))

                # TODO - Check ig it is necessary
                lb = min(lower_bound,upper_bound)
                ub = max(lower_bound,upper_bound)

                lb_function = self.flash_equation(lb, zi, Ki)
                ub_function = self.flash_equation(ub, zi, Ki)

                if lb_function*ub_function < 0:
                    vi = optimize.brentq(self.flash_equation, a=lb, b=ub, args=(zi, Ki))
                else:
                    resp = optimize.root(self.flash_equation, x0=(lb+ub)/2, args=(zi, Ki))
                    vi = resp.x

                x_flatten[:,i] = zi / (1 + vi * (Ki - 1))
                y_flatten[:,i] = x_flatten[:,i] * Ki

                if np.isnan(np.sum(x_flatten)) or np.isnan(np.sum(x_flatten)):
                    b = 0

                # Taking x and y positives and normalized
                #x_flatten = abs(x_flatten)/np.sum(abs(x_flatten),axis=0)
                #y_flatten = abs(y_flatten)/np.sum(abs(y_flatten),axis=0)

                #Writing vi to v array and limiting it
                v[i] = vi

            # Expanding shape
            v = np.reshape(v, temperature.shape)
            x = np.reshape(x_flatten, composition.shape)
            y = np.reshape(y_flatten, composition.shape)

            if np.isnan(np.sum(x)) or np.isnan(np.sum(y)):
                a = 0 

            eos_liq = self.stream_liq.calculate_eos(x, temperature, pressure)
            eos_vap = self.stream_vap.calculate_eos(y, temperature, pressure)

        
            # Calculating residue
            factor = (x*eos_liq.phi)/(y*eos_vap.phi)
            diff = np.sqrt(np.sum((factor - 1)**2, axis=0))

            # Incrementing K
            K *= factor

            K_flatten = K

            # Incrementing count
            cont += 1

        for i in range(len(temperature)):
            if v[i] >= 1:
                v[i] = 1-1e-10
                y_flatten[:,i] = composition[:,i]
                #x_flatten[:,i] = composition[:,i]
            elif v[i] < 0:
                v[i] = 1e-10
                x_flatten[:,i] = composition[:,i]
                #y_flatten[:,i] = composition[:,i]
            flowrate_L_flatten[:,i] = np.diag(stream.components_properties.MM*1e-3) @ (x_flatten[:,i] * total_molar_flowrate_flatten[i] * (1-v[i]))
            flowrate_V_flatten[:,i] = np.diag(stream.components_properties.MM*1e-3) @ (y_flatten[:,i] * total_molar_flowrate_flatten[i] * v[i])

        self.stream_liq.specify(flowrate_L_flatten, temperature, pressure, 1-v)
        self.stream_vap.specify(flowrate_V_flatten, temperature, pressure, v) 

        info = [cont, diff]

        return self.stream, self.stream_liq, self.stream_vap, K, v, info
