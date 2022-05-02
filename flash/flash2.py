from abc import ABC, abstractmethod
import numpy as np
from auxiliar.stream import Stream


class Flash2(ABC):
    """
    Two-phase flash abstract class
    """

    def __init__(self, stream: Stream, tol=1e-6, contmax=200):
        """
        Two-phase flash constructor
        :param stream:
        :param stream_liq:
        :param stream_vap:
        :param tol:
        :param contmax:
        """

        # Storing attributes
        self.tol = tol
        self.contmax = contmax

        # Stream instance
        self.stream = stream

        # Each phase stream
        self.stream_liq = Stream(stream.components_properties)
        self.stream_vap = Stream(stream.components_properties)

        # Number of components
        self.N = self.stream.components_properties.N


    def __call__(self, flowrate, temperature, pressure, K0=None):

        self.stream.specify(flowrate, temperature, pressure)

        # Calculate initial estimate
        if K0 is None:
            K = self.calculate_wilson(temperature, pressure)
        else:
            K = K0

        return self.solve(K)


    @abstractmethod
    def solve(self, K0):
        """
        This method solves the flash. It must be re-written by each concrete flash class
        :param z: global molar composition of self.stream
        :param T: temperature in K
        :param P: pressure in Bar
        :param K0:
        :return:
        """
        return self.stream, self.stream_liq, self.stream_vap, K0

    def calculate_wilson(self, temperature, pressure):
        """

        :param pressure: pressure in Bar
        :param temperature: temperature in K
        :return:
        """

        # Reading properties
        w = self.stream.components_properties.w
        Tc = self.stream.components_properties.Tc
        Pc = self.stream.components_properties.Pc


        # Defining the shape of the K array based on on the number and on
        K_shape = [self.N,] + list(temperature.shape)
        K = np.empty(K_shape)

        for i, (wi, Tci, Pci) in enumerate(zip(w, Tc, Pc)):
            K[i] = Pci / pressure * np.exp((5.37 * (1 + wi) * (1 - Tci / temperature)).astype(float))

        return K
