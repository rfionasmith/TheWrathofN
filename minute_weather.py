# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


f_name = "VEX_Feb2015_t.txt"

data = np.loadtxt(f_name,usecols=(3,4,5,6,7,8,9,10,11),skiprows=8) #add 11 for all other files
time = np.loadtxt(f_name,usecols=0,skiprows=8)
time = np.mod(time,86400)/3600

night_mask = np.logical_and(time>4,time<16)

for i in range(1,9):
    for j in range(1,9):
        if i == j:
            continue
        if i ==6 or j == 6:
            continue
        antA = data[night_mask,i]
        antB = data[night_mask,j]

        plt.hist2d(antA,antB,bins=201,range=[[-10.05,10.05],[-10.05,10.05]],cmap=plt.cm.plasma)
        x=np.linspace(-10,10,201)

        z = np.polyfit(antA,antB,1)
        p = np.poly1d(z)

        off_fit = antB - p(antA)
        lower = np.percentile(off_fit,15.866)
        upper = np.percentile(off_fit,84.134)
        print(lower,upper)
        
        plt.plot(x,p(x),'-y',lw=0.5)
        plt.plot(x,p(x)-abs(lower),'--y',lw=0.5)
        plt.plot(x,p(x)+abs(upper),'--y',lw=0.5)
        
        plt.title("VEX Feb 2015 "+str(i)+" vs "+str(j))
        plt.xlabel("Temperature (degC) of Ant "+str(i),fontsize='xx-small')
        plt.ylabel("Temperature (degC) of Ant "+str(j),fontsize='xx-small')
        plt.xticks(fontsize='xx-small')
        plt.yticks(fontsize='xx-small')
        
        plt.show()
'''        
plt.hist2d(antA,antB,bins=201,range=[[-10.05,10.05],[-10.05,10.05]],cmap=plt.cm.plasma)
x=np.linspace(-10,10,201)

#plt.hist2d(antA,antB,bins=161,range=[[616.95,633.05],[616.95,633.05]],cmap=plt.cm.plasma)
#x=np.linspace(617,633,161)

#plt.hist2d(antA,antB,bins=100,range=[[-0.5,99.5],[-0.5,99.5]],cmap=plt.cm.plasma)
#x=np.linspace(0,99,100)

'''
