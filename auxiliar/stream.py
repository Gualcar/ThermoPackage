from typing import Tuple
import numpy as np

from scipy import optimize

from .phase_properties import PhaseProperties
from .components_properties import ComponentsProperties
from .phases import PhaseName
from eos.eos import EquationOfState

from correlations.viscosity import calculate_viscosity

#from .flash.flash2_via_root import Flash2ViaNR

class Stream(object):
    """
    Stream class
    """

    def __init__(
        self,
        components_properties: ComponentsProperties,
    ):
        """
        Stream constructor
        :param components_properties: component properties
        :param phases: tuple with the phases
        """

        # Storing values
        self.components_properties = components_properties

        # Storing nones
        self.composition = None
        self.T = None
        self.P = None
        self.phase = None
        self.properties = None
        self.eos = EquationOfState(self)

        self.molar_volume_fraction = None


    def calculate_molar_flowrate(self, flowrate):
        return np.diag(1/self.components_properties.MM)@flowrate*1e3

    @staticmethod
    def calculate_total_flowrate(molar_flowrate):
        return np.sum(molar_flowrate,axis=0)

    @staticmethod
    def calculate_molar_composition(molar_flowrate, total_molar_flowrate):
        return molar_flowrate@np.diag(1/total_molar_flowrate)


    def specify(self,
                flowrate: np.ndarray,
                temperature: np.ndarray,
                pressure: np.ndarray,
                molar_volume_fraction = None
                ):
        """
        This method defines the specifications and calculates the properties (solver the equation of state)
        :param flowrate:
        :param temperature:
        :param pressure:
        :param phase:
        :return:
        """

        molar_flowrate = self.calculate_molar_flowrate(flowrate)
        total_flowrate = self.calculate_total_flowrate(flowrate)
        total_molar_flowrate = self.calculate_total_flowrate(molar_flowrate)
        molar_composition = self.calculate_molar_composition(molar_flowrate, total_molar_flowrate)

        # Storing values
        self.T = temperature
        self.P = pressure
        self.flowrate = flowrate
        self.total_flowrate = total_flowrate
        self.molar_flowrate = molar_flowrate
        self.total_molar_flowrate = total_molar_flowrate
        self.composition = molar_composition
        self.properties = self.calculate_eos(self.composition, temperature, pressure)

        #After the properties calculation the volume flowrate can be calculated
        self.volume_flowrate = flowrate/self.properties.density
        self.total_volume_flowrate = self.calculate_total_flowrate(self.volume_flowrate)

        self.molar_volume_fraction = molar_volume_fraction


    def calculate_eos(self, composition, temperature, pressure):
        return self.eos(composition, temperature, pressure)


    def __repr__(self):
        return "{}(T: {}, P: {}, phase: {})".format(self.__class__, self.T, self.P, self.phase,)
