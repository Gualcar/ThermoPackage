import numpy as np
from auxiliar.phases import PhaseName
from auxiliar.phase_properties import PhaseProperties
from .cubic_equation import cubic_equation
from correlations.viscosity import calculate_viscosity, calculate_viscosity_simple, calculate_viscosity_ali 
from correlations.surface_tension import calculate_surface_tension

from auxiliar.root_finding import roots_n3

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from auxiliar.stream import Stream
    from auxiliar.components_properties import ComponentsProperties

# =========================================
# Equantion of state proposed by Abbas 2016
# Thermodynamis and Applications in Hydrocarbon Energy Production
#==========================================


class EquationOfState:

    @staticmethod
    def composition_weighted_sum(composition: np.ndarray, value: np.ndarray):
        """
        Weighted sum based on composition
        :param composition:
        :param value:
        :return:
        """
        out_value = np.zeros_like(composition[0])
        for composition_i, value_i in zip(composition, value):
            out_value += composition_i * value_i
        return out_value

    @staticmethod
    def expand_property(composition: np.ndarray, property: np.ndarray):
        """
        Expands a property that has the dimension of composition to cover the dimension of the composition array
        :param composition:
        :param property:
        :return:
        """
        new_property = np.empty_like(composition)
        for i, composition_i in enumerate(composition):
            new_property[i] = property[i] * np.ones_like(composition_i)
        return new_property

    @staticmethod
    def build_aij(ap, kij):
        n = len(ap)
        aij = np.empty((n,n))
        for i in range(n):
            for j in range(n):
                aij[i,j] = ((1 - kij[i,j])*(ap[i]**0.5)*(ap[j]**0.5))**2
        return aij

    def __init__(self, stream: "Stream"):
        """
        Constructs the EOS
        :param phase:
        """
        # Reading properties
        props = stream.components_properties
        self.stream = stream

        # EOS parameters
        self.eps = 1 + np.sqrt(2)
        self.sig = 1 - np.sqrt(2)
        self.omg = 0.07780
        self.psi = 0.45724
        self.Zc = 0.30740
        self.k = 0.37464 + 1.54226 * props.w - 0.26992 * props.w ** 2

        self.bp = (self.omg*8.314*1e-5*props.Tc)/props.Pc

        # TODO - It must be improved in the furure. Possibly it will migrate to the stream.components_properties
        self.kij = np.zeros([props.N, props.N])


    def __call__(self, composition, temperature, pressure):

        stream = self.stream

        # Read the component properties
        props = stream.components_properties


        #Normalize the composition before do the calculations
        composition = abs(composition)/np.sum(abs(composition), axis=0)

        alfa = np.empty_like(composition)
        alfa_dT = np.empty_like(composition)
        for i in range(stream.components_properties.N):
            Tri = temperature/ props.Tc[i]
            alfa[i,:] = (1 + self.k[i] * (1 - Tri ** (0.5))) ** 2
            alfa_dT[i,:] = -self.k[i] * (1 + self.k[i] * (1 - Tri ** (0.5))) / (np.sqrt(temperature*props.Tc[i]))
        self.alfa = alfa
        self.alfa_dT = alfa_dT

        # (1) Compressibility factor calculation

        #Using bar instead of Pa

        # TODO - This part of the code MUST BE IMPROVED. Many variables without meaningful name.
        # TODO - R with different units is not OK. Must standardize.

        R = 8.314*1e-5

        self.bm = self.composition_weighted_sum(composition, self.bp)

        self.ap = np.empty_like(composition)
        for i in range(0, props.N):
            self.ap[i] = ((self.psi * (R * props.Tc[i]) ** 2) / props.Pc[i]) * self.alfa[i]

        self.am = np.zeros_like(composition[0])
        for i in range(0, props.N):
            for j in range(0, props.N):
                self.am += composition[i]*composition[j]*(self.ap[i]*self.ap[j])**0.5*(1-self.kij[i,j])

        self.A = self.am * pressure / (R * temperature) ** 2
        self.B = self.bm * pressure / (R * temperature)

        # Calculating Z in a loop
        # We will take de compressibility with me minimum Gibbs Energy
        # Page 362
        # TODO - Must be improved for better performance. Numba candidate
        self.Z = np.empty_like(temperature)
        aij = []
        for i, (Ai, Bi,) in enumerate(zip(self.A, self.B,)):
            aij.append(self.build_aij(self.ap[:,i], self.kij))
            Z_i = cubic_equation(Ai, Bi, pressure, temperature, composition)
            self.Z[i] = Z_i

        # EquationOfState initialy only gives me the compressibility
        # I don't know how many phases there are in the system

        # Instantiate the phase properties
        properties = PhaseProperties()

        # Storing compressibility
        properties.set_compressibility(self.Z)

        # (2) Density calculation

        mix_MM = self.composition_weighted_sum(composition, props.MM)
        molrho = pressure / (self.Z * R * temperature)
        rho = molrho * mix_MM * 1e-3
        properties.set_density(rho)
        properties.set_molar_volume(1/molrho)

        x_aij = np.empty_like(composition)
        for i in range(len(temperature)):
            x_aij[:,i] = composition[:,i]@aij[i] 

        # (3) phi calculation
        self.lnphi = np.empty_like(composition)
        for i in range(len(temperature)):
            self.lnphi[:,i] = (self.bp / self.bm[i]) * (self.Z[i] - 1) - np.log(self.Z[i] - self.B[i]) + (self.A[i]/(2.82*self.B[i]))*(2*x_aij[:,i]/self.am[i] - self.bp/self.bm[i])*np.log((self.Z[i] + 2.414*self.B[i])/(self.Z[i] - 0.414*self.B[i]))
        self.phi = np.exp(np.array(self.lnphi, dtype=float))
        properties.set_lnphi(self.lnphi)
        properties.set_phi(self.phi)

        # (4) Viscosity calculation
        #pure_molar_density = self.calculate_pure_molar_density(temperature, pressure)
        #methane_critical_molar_desity = self.calculate_pure_molar_density(props.Tc[0]*np.ones_like(temperature), props.Pc[0]*np.ones_like(pressure))
        viscosity = calculate_viscosity_simple(rho)
        #viscosity_ali = calculate_viscosity_ali(composition, temperature, pressure, molrho, rho, props, self)
        properties.set_viscosity(viscosity)

        # (5) Surface Tension calculation
        surface_tension = calculate_surface_tension(composition, temperature, pressure, props)
        properties.set_surface_tension(surface_tension)

        # (6) enthalpy calculation

        #Cálculo de outras propriedades termodinâmicas
        # Parâmetros u e w para a equação de Peng-Robinson
        _u = 2
        _w = -1
        _s0 = np.zeros_like(temperature)
        s0_teste = np.zeros_like(temperature)
        for i in range(props.N):
            _s1 = np.zeros_like(temperature)
            s1_teste = np.zeros_like(temperature)
            for j in range(props.N):
                _s1 += composition[i,:] * composition[j,:] * (1 - self.kij[i, j]) * (
                            self.psi * props.Tc[i] * props.Tc[j] * (R ** 2) / (
                        np.sqrt(props.Pc[i] * props.Pc[j]))) * (0.5 * (1 / np.sqrt(self.alfa[i,:] * self.alfa[j,:])) * (
                            self.alfa[i,:] * self.alfa_dT[j,:] + self.alfa[j,:] * self.alfa_dT[i,:]))
                s1_teste += composition[i,:] * composition[j,:] * (1 - self.kij[i, j]) * (self.k[j]*np.sqrt(self.ap[i]*props.Tc[j]/props.Pc[j]) + self.k[i]*np.sqrt(self.ap[j]*props.Tc[i]/props.Pc[i]))
            _s0 += _s1
            s0_teste += s1_teste
        self._dadT = _s0
        self.dadT_teste = - R * s0_teste * 0.5*np.sqrt(0.45724/temperature)

        self.HR = ((temperature * self._dadT - self.am) / (self.bm * 2 *np.sqrt(2))) * np.log(
            (self.Z + self.B * (1 + np.sqrt(2))) / (
                        self.Z + self.B * (1 - np.sqrt(2)))) + R * temperature * (self.Z - 1)

        teste1 = -9.99223e-9
        self.teste = ((temperature*teste1 - self.am)/(self.bm*np.sqrt(_u**2-4*_w)))*np.log((2*self.Z + self.B*(_u + np.sqrt(_u**2-4*_w)))/(2*self.Z + self.B*(_u - np.sqrt(_u**2-4*_w)))) + R*temperature*(self.Z - 1)
        self.teste = (self.teste / mix_MM) * 1e8
        self.HR = (self.HR / mix_MM) * 1e8
        # deltaHform = props.deltaHform/props.MM

        igA = self.expand_property(composition, props.igA)
        igB = self.expand_property(composition, props.igB)
        igC = self.expand_property(composition, props.igC)
        igD = self.expand_property(composition, props.igD)
        hA = self.expand_property(composition, props.hA)
        hB = self.expand_property(composition, props.hB)
        hC = self.expand_property(composition, props.hC)
        hD = self.expand_property(composition, props.hD)
        hE = self.expand_property(composition, props.hE)
        hF = self.expand_property(composition, props.hF)
        deltaHform = self.expand_property(composition, props.deltaHform)
        self.ideal_gas_cp = (igA + igB * temperature + igC * temperature ** 2 + igD * (temperature ** -2)) * R *1e5

        self.ideal_gas_enthalpy_hysys = hA + hB*temperature + hC*temperature**2 + hD*temperature**3 + hE*temperature**4 + hF*temperature**5
        self.ideal_gas_mix_enthalpy_hysys = (self.composition_weighted_sum(stream.flowrate/np.sum(stream.flowrate, axis=0), self.ideal_gas_enthalpy_hysys))*1e3

        self.ideal_gas_mix_cp = self.composition_weighted_sum(composition,self.ideal_gas_cp)
        properties.set_ideal_cp(self.ideal_gas_mix_cp)
        
        self.ideal_gas_enthalpy = (igA * (temperature - 298.15) + (igB / 2) * (temperature ** 2 - 298.15 ** 2) + (igC / 3) * (temperature ** 3 - 298.15 ** 3) + igD * (1 / temperature - 1 / 298.15)) * R*1e5 + deltaHform
        self.ideal_gas_enthalpy = (self.ideal_gas_enthalpy/props.MM.reshape(-1,1))*1e3

        self.ideal_gas_mix_enthalpy = (self.composition_weighted_sum(stream.flowrate/np.sum(stream.flowrate, axis=0), self.ideal_gas_enthalpy))
        enthalpy = self.HR + self.ideal_gas_mix_enthalpy

        # enthalpy_hysys = self.HR + self.ideal_gas_mix_enthalpy_hysys
        # properties.set_enthalpy(enthalpy_hysys)
        properties.set_enthalpy(enthalpy)


        # Activate if run RNA is activated

        return properties


    def calculate_pure_molar_density(self, temperature, pressure):

        stream = self.stream

        # Read the component properties
        props = stream.components_properties

        # Each component will be calculated isolated

        id_composition = np.zeros([len(props.Tc),len(props.Tc)])
        for i in range(len(props.Tc)):
            id_composition[i,i] = 1

        pure_mass_density = np.zeros([len(props.Tc),len(temperature)])
        pure_molar_density = np.zeros([len(props.Tc),len(temperature)])

        for comp in range(len(props.Tc)):

            composition = np.tile(id_composition[:,comp].reshape(-1,1), len(temperature))

            alfa = np.empty_like(composition)
            alfa_dT = np.empty_like(composition)
            for i in range(stream.components_properties.N):
                Tri = temperature/ props.Tc[i]
                alfa[i,:] = (1 + self.k[i] * (1 - Tri ** (0.5))) ** 2
                alfa_dT[i,:] = -self.k[i] * (1 + self.k[i] * (1 - Tri ** (0.5))) / (np.sqrt(temperature*props.Tc[i]))
            self.alfa = alfa
            self.alfa_dT = alfa_dT

            # (1) Compressibility factor calculation

            #Using bar instead of Pa

            # TODO - This part of the code MUST BE IMPROVED. Many variables without meaningful name.
            # TODO - R with different units is not OK. Must standardize.

            R = 8.314*1e-5

            bm = self.composition_weighted_sum(composition, self.bp)

            ap = np.empty_like(composition)
            for i in range(0, props.N):
                ap[i] = ((self.psi * (R * props.Tc[i]) ** 2) / props.Pc[i]) * self.alfa[i]

            am = np.zeros_like(composition[0])
            for i in range(0, props.N):
                for j in range(0, props.N):
                    am += composition[i]*composition[j]*(ap[i]*ap[j])**0.5*(1-self.kij[i,j])

            A = am * pressure / (R * temperature) ** 2
            B = bm * pressure / (R * temperature)

            # Calculating Z in a loop
            # We will take de compressibility with me minimum Gibbs Energy
            # Page 362
            # TODO - Must be improved for better performance. Numba candidate
            Z = np.empty_like(temperature)
            aij = []
            for i, (Ai, Bi,) in enumerate(zip(A, B,)):
                aij.append(self.build_aij(ap[:,i], self.kij))
                Z_i = cubic_equation(Ai, Bi, pressure, temperature, composition)
                Z[i] = Z_i

            # EquationOfState initialy only gives me the compressibility
            # I don't know how many phases there are in the system

            # (2) Density calculation

            mix_MM = self.composition_weighted_sum(composition, props.MM)
            molrho = pressure / (Z * R * temperature)
            rho = molrho * mix_MM * 1e-3

            pure_mass_density[comp,:] = rho
            pure_molar_density[comp,:] = molrho

        return pure_mass_density, pure_molar_density