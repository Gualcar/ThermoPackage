import numpy as np

from auxiliar.components_properties import ComponentsProperties
from auxiliar.stream import Stream
from flash.flash2_negative import Flash2ViaSS
from flash.flash2_normal import Flash2ViaSS_normal

#=============  Defining the inputs  ================

components = ['C1', 'C2', 'nC8'] # see alias on 'data' archives
composition = np.array([0.1, 0.3, 0.6]) # molar fraction

each_pressure = np.linspace(10, 50, 5) # bar
each_temperature = np.linspace(298, 333, 5) # K

#Create a combination of pressure and temperature samples
pressures, temperatures = np.meshgrid(each_pressure, each_temperature)
pressures = pressures.flatten()
temperatures = temperatures.flatten()
points = len(pressures)

#=============  Create the objects  ================

#Read the components properties
components_properties = ComponentsProperties(components)
stream = Stream(components_properties)

#Create the flash object
flash_negative = Flash2ViaSS(stream)
flash_normal = Flash2ViaSS_normal(stream)

#Transform composition to accept the pressure and temperature vector
composition = np.tile(composition.reshape(-1,1),len(temperatures))

#=============  Run the flash  ===================

#Generate the samples
stream, liquid_stream, vapour_stream, K, v, info = flash_normal(composition, temperatures, pressures)

#=============  Read the results  =================

#Access each phase properties through the flash call outputs
print(v)
print(info)