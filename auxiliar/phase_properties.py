class PhaseProperties:
    """
    Phase Properties to be calculated by the equation of state call
    """

    def __init__(self):

        # Initialize Nones
        self.enthapy = None
        self.density = None
        self.viscosity = None
        self.molar_volume = None
        self.compressibility = None
        self.phi = None
        self.lnphi = None
        self.surface_tension = None
        self.ideal_cp = None


    def set_compressibility(self, value):
        """
        Compressibility setter
        :param value:
        :return:
        """
        self.compressibility = value

    def set_density(self, value):
        """
        Density setter
        :param value:
        :return:
        """
        self.density = value

    def set_molar_volume(self, value):
        """
        Molar colume setter
        :param value:
        :return:
        """
        self.molar_volume = value

    def set_enthalpy(self, value):
        """
        Enthalpy setter
        :param value:
        :return:
        """
        # TODO - Must be reviewed
        self.enthalpy = value

    def set_viscosity(self, value):
        """
        Viscosity setter
        :param value:
        :return:
        """
        # TODO - Must be reviewed
        self.viscosity = value

    def set_phi(self, value):
        """
        Phi setter
        :param value:
        :return:
        """
        self.phi = value

    def set_lnphi(self, value):
        """
        Phi setter
        :param value:
        :return:
        """
        self.lnphi = value

    def set_surface_tension(self, value):
        """
        Surface Tension setter
        :param value:
        :return:
        """
        self.surface_tension = value

    def set_ideal_cp(self, value):
        """
        .ideal gas cp
        :param value:
        :return:
        """
        self.ideal_cp = value