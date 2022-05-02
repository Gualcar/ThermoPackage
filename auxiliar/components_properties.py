import pandas as pd
from typing import Tuple
import numpy as np

PURE_PROPERTIES_DATABASE_FILE = 'data/pure_comp_prop.csv'
PURE_PROPERTIES_DATABASE_SEPARATOR = ';'

IDEAL_GAS_PROPERTIES_DATABASE_FILE = 'data/ideal_gas_prop.csv'
IDEAL_GAS_PROPERTIES_DATABASE_SEPARATOR = ','

IDEAL_GAS_PROPERTIES_DATABASE_HYSYS_FILE = 'data/ideal_gas_prop_hysys.csv'
IDEAL_GAS_PROPERTIES_DATABASE_HYSYS_SEPARATOR = ','

class ComponentsProperties:

    def __init__(self, components: Tuple[str,...]):
        """
        Read, parses and filter the component properties
        :param components: tuple of component name
        """

        # Number of components
        self.N = len(components)

        # Pure component properties
        self.pure_properties = ComponentsProperties.get_pure_properties(components)

        # Pure component ideal gas properties
        self.ideal_gas_properties = ComponentsProperties.get_ideal_gas_properties(components)

        # Pure component ideal gas properties
        self.ideal_gas_properties_hysys = ComponentsProperties.get_ideal_gas_properties_hysys(components)

    @staticmethod
    def filter_components(df, components: Tuple[str,...]):
        """
        Filter the dataframe based on component abrev
        :param df:
        :param components:
        :return:
        """

        df_abrev = df.iloc[:, 1]
        list_abrev = list(df_abrev.values)
        comp_selec = [list_abrev.index(i) for i in components]
        df_selec = df.iloc[comp_selec, :]

        return df_selec


    @staticmethod
    def get_pure_properties(components: Tuple[str,...]):
        """
        Read, parses and filter the pure properties database
        :param components: tuple of component name
        :return:
        """

        df_prop = pd.read_csv(
            PURE_PROPERTIES_DATABASE_FILE,
            sep=PURE_PROPERTIES_DATABASE_SEPARATOR,
            dtype={
                "Specie": str,
                "Abrev": str,
                "Molar_Mass": np.float64,
                "Acentric": np.float64,
                "Tc_K": np.float64,
                "Pc_bar": np.float64,
                "Zc": np.float64,
                "Vc_cm3mol": np.float64,
                "Tn_K": np.float64,
                "deltaHform_298": np.float64,
            },
        )

        df_selec = ComponentsProperties.filter_components(df_prop, components)

        return df_selec

    @staticmethod
    def get_ideal_gas_properties(components: Tuple[str,...]):
        """
        Read, parses and filter the ideal gas stat properties database
        :param components:
        :return:
        """

        df_prop = pd.read_csv(
            IDEAL_GAS_PROPERTIES_DATABASE_FILE,
            sep=IDEAL_GAS_PROPERTIES_DATABASE_SEPARATOR,
            dtype={
                "Specie": str,
                "Abrev": str,
                "A_1": np.float64,
                "B_10_3": np.float64,
                "C_10_6": np.float64,
                "D_10_m5": np.float64,
            }
        )

        df_prop["B_1"] = df_prop["B_10_3"] * 1e-3
        df_prop["C_1"] = df_prop["C_10_6"] * 1e-6
        df_prop["D_1"] = df_prop["D_10_m5"] * 1e5

        df_selec = ComponentsProperties.filter_components(df_prop, components)

        return df_selec

    @staticmethod
    def get_ideal_gas_properties_hysys(components: Tuple[str,...]):
        """
        Read, parses and filter the ideal gas stat properties database
        :param components:
        :return:
        """

        df_prop = pd.read_csv(
            IDEAL_GAS_PROPERTIES_DATABASE_HYSYS_FILE,
            sep=IDEAL_GAS_PROPERTIES_DATABASE_HYSYS_SEPARATOR,
            dtype={
                "Specie": str,
                "Abrev": str,
                "A": np.float64,
                "B": np.float64,
                "C": np.float64,
                "D": np.float64,
                "E": np.float64,
                "F": np.float64,
            }
        )

        df_selec = ComponentsProperties.filter_components(df_prop, components)

        return df_selec

    @property
    def comp(self) -> str:
        return self.pure_properties["Abrev"].values

    @property
    def Pc(self) -> float:
        return self.pure_properties["Pc_bar"].values

    @property
    def Tc(self) -> float:
        return self.pure_properties["Tc_K"].values

    @property
    def Tn(self) -> float:
        return self.pure_properties["Tn_K"].values

    @property
    def Vc(self) -> float:
        return self.pure_properties["Vc_cm3mol"].values

    @property
    def Zc(self) -> float:
        return self.pure_properties["Zc"].values

    @property
    def w(self) -> float:
        return self.pure_properties["Acentric"].values

    @property
    def MM(self) -> float:
        return self.pure_properties["Molar_Mass"].values

    @property
    def deltaHform(self) -> float:
        return self.pure_properties["deltaHform_298"].values

    @property
    def igA(self) -> float:
        return self.ideal_gas_properties["A_1"].values

    @property
    def igB(self) -> float:
        return self.ideal_gas_properties["B_1"].values

    @property
    def igC(self) -> float:
        return self.ideal_gas_properties["C_1"].values

    @property
    def igD(self) -> float:
        return self.ideal_gas_properties["D_1"].values

    @property
    def hA(self) -> float:
        return self.ideal_gas_properties_hysys["A"].values

    @property
    def hB(self) -> float:
        return self.ideal_gas_properties_hysys["B"].values

    @property
    def hC(self) -> float:
        return self.ideal_gas_properties_hysys["C"].values

    @property
    def hD(self) -> float:
        return self.ideal_gas_properties_hysys["D"].values
    
    @property
    def hE(self) -> float:
        return self.ideal_gas_properties_hysys["E"].values
    
    @property
    def hF(self) -> float:
        return self.ideal_gas_properties_hysys["F"].values
