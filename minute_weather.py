# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt


f_name = "SC_Dec2020_t.txt"

data = np.loadtxt(f_name,usecols=(3,4,5,6,7,8,9,10,11),skiprows=8) #add 11 for all other files
time = np.loadtxt(f_name,usecols=0,skiprows=8)
time = np.mod(time,86400)/3600

night_mask = np.logical_and(time>4,time<16)

#for i in range(0,8):
#    plt.plot(time, data[:,i],'.',label=str(i+1))
#plt.title(f_name)
#plt.legend(fontsize='x-small')
#plt.show()

antA = data[night_mask,1]
antB = data[night_mask,2]

plt.hist2d(antA,antB,bins=151,range=[[-5.05,10.05],[-5.05,10.05]],cmap=plt.cm.plasma)
x=np.linspace(-5,10,151)

#plt.hist2d(antA,antB,bins=161,range=[[616.95,633.05],[616.95,633.05]],cmap=plt.cm.plasma)
#x=np.linspace(617,633,161)

#plt.hist2d(antA,antB,bins=100,range=[[-0.5,99.5],[-0.5,99.5]],cmap=plt.cm.plasma)
#x=np.linspace(0,99,100)

z = np.polyfit(antA,antB,1)
p = np.poly1d(z)
print(p)

off_line = antB - p(antB)
print(off_line)
lower = np.percentile(off_line,25.)
upper = np.percentile(off_line,75.)
print(lower,upper)

plt.plot(x,p(x),'--y',lw=0.5)
plt.plot(x,p(x)+lower,lw=0.5)

plt.xlabel("AntA = 1",fontsize='xx-small')
plt.ylabel("AntB = 2",fontsize='xx-small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.colorbar()
plt.show()

#plt.plot(antA)
plt.plot(antA,antB,'.',lw=.1,alpha=.1)
plt.plot(x,p(x),'--',lw=0.5)
#plt.plot(antB,p(antB)-0.5)
#plt.plot(p(antB))
#plt.plot(off_line)
plt.show()

