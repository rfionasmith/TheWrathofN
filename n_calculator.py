# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:54:55 2021

@author: rfts1
"""
import numpy as np
import matplotlib.pyplot as plt

# pAtm is pressure in mbar
# tAtm is temp in deg C
# relHumidity is RH in %
# refracVals is N in refractivity units (where n = 1 + (N*1e-6))


def get_N(temp,rh):
    
    pAtm = 621.0
    tAtm = temp
    relHumidity = rh


    efWater =  1 + ((1e-4)*(7.2 + (pAtm*(0.00320 + ((5.9e-7)*(tAtm)**2)))))

    satWaterVapor = efWater*6.1121*np.exp(((18.678-((tAtm)/234.5))*(tAtm))/(tAtm+257.14))

    waterVapPressure = (satWaterVapor*relHumidity/100)

    refracVals = (77.6*pAtm/(tAtm+273.15)) - 6*(waterVapPressure/(tAtm+273.15)) + 3.75e5*(waterVapPressure/((tAtm+273.15)**2))

    return(refracVals)


dry = get_N(0.0,15)
wet = get_N(0.0,85)

delta_RH = np.array([2.2,2.8,3.0,3.5,4.1])
delta_T = np.array([0.6,0.7,0.8,0.9,1.1])

configs = ["SUB","COM","EX1","EX2","VEX"]
configs = [30,50,100,200,500]

path_humidity = []
for idx in range(0,5):
    path = (get_N(0.0,15.0+delta_RH[idx])-dry)/10.
    path_humidity.append(path)
    
path_humidity_wet = []
for idx in range(0,5):
    path = (get_N(0.0,85.0+delta_RH[idx])-wet)/10.
    path_humidity_wet.append(path)

path_temp_dry = []
for idx in range(0,5):
    path = abs(get_N(delta_T[idx],15.0)-dry)/10.
    path_temp_dry.append(path)
print(path_temp_dry)    

path_temp_wet = []
for idx in range(0,5):
    path = (get_N(delta_T[idx],85.0)-wet)/10.
    path_temp_wet.append(path)
print(path_temp_wet)
    
plt.plot(configs,path_humidity,'.-',label = 'Humidity')
#plt.semilogx(configs,path_humidity_wet,'.-')
plt.plot(configs,path_temp_dry,'.-',label = 'Temperature (dry)')
plt.plot(configs,path_temp_wet,'.-',label = 'Temperature (wet)')
plt.legend(fontsize = 'xx-small')
plt.ylabel('Path Length Difference (mm)')
plt.xlabel('Max Baseline (m)')
plt.show()    