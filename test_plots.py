# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:19:17 2021

@author: rfts1
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

#pp = PdfPages('result_plots.pdf')
data = pd.read_csv('results.txt')
configs = [0,30,50,100,200,500,1000]


distance = data['N-S']
values = data['t_sigma']
'''
sc = np.logical_and(distance>0,distance<=30)
co = np.logical_and(distance>30,distance<=50)
ex1 = np.logical_and(distance>50,distance<=150)
vex = np.logical_and(distance>150,distance<=1000)

plt.semilogx(distance,values,'.',markersize=2)
plt.grid(True,which='both')

bins = [np.median(values[sc]),np.median(values[co]),np.median(values[ex1]),np.median(values[vex])]
lower = [np.percentile(values[sc],15.866),np.percentile(values[co],15.866),np.percentile(values[ex1],15.866),np.percentile(values[vex],15.866)]
upper = [np.percentile(values[sc],84.134),np.percentile(values[co],84.134),np.percentile(values[ex1],84.134),np.percentile(values[vex],84.134)]
errors = [lower,upper]
positions = [np.median(distance[sc]),np.median(distance[co]),np.median(distance[ex1]),np.median(distance[vex])]
#plt.errorbar(positions,bins,yerr=errors,fmt='.',capsize=1.5)
#plt.xlim([1,600])
#plt.show()


for i in range(1,6):
    plt.subplot(2,3,i)
    if i < 4:
        plt.title(data.columns.values[i+4],fontsize='x-small')
        plt.semilogx(data.iloc[:,i+4],values,'.b',markersize=1,alpha=0.7)
        distance = data.iloc[:,i+4]
        sc = np.logical_and(distance>0,distance<=30)
        co = np.logical_and(distance>30,distance<=50)
        ex = np.logical_and(distance>50,distance<=150)
        vex = np.logical_and(distance>150,distance<=1000)
    else:
        plt.xlabel(data.columns.values[i+5],fontsize='x-small')
        plt.semilogx(data.iloc[:,i+5],values,'.b',markersize=1,alpha=0.7)
        distance = data.iloc[:,i+5]
        sc = np.logical_and(distance>0,distance<=30)
        co = np.logical_and(distance>30,distance<=50)
        ex = np.logical_and(distance>50,distance<=150)
        vex = np.logical_and(distance>150,distance<=1000)
    
    bins = np.array([np.median(values[sc]),np.median(values[co]),np.median(values[ex]),np.median(values[vex])])
    lower = np.array([np.percentile(values[sc],15.866),np.percentile(values[co],15.866),np.percentile(values[ex],15.866),np.percentile(values[vex],15.866)])
    upper = np.array([np.percentile(values[sc],84.134),np.percentile(values[co],84.134),np.percentile(values[ex],84.134),np.percentile(values[vex],84.134)])
    errors = [abs(bins-lower),abs(bins-upper)]
    positions = [np.median(distance[sc]),np.median(distance[co]),np.median(distance[ex]),np.median(distance[vex])]
    plt.errorbar(positions,bins,yerr=errors,fmt='.k',capsize=1.5)
    print(bins)
    plt.grid(True,which='both')
    #plt.yticks(np.arange(0,7,step=1),fontsize='xx-small')
    plt.xticks(fontsize='xx-small')
               
    #for xc in configs:
     #   plt.axvline(x=xc,c='black',lw=1)    

    plt.xlim([1,600])
plt.subplot(2,3,6)
#plt.boxplot(data['h'],sym='.',widths=1.5,positions=[1])
plt.axis('off')
plt.xlim([0,10])
plt.suptitle('Humidity Offsets')
plt.show()
'''
#pp.close()
plt.figure(figsize=(11,2.1))
plt.suptitle("Temperature Offsets")
for i in range(1,6):
    plt.subplot(1,5,i)
    #plt.yticks(np.arange(0,5,step=1))
    if i ==5:
        i+=1
    if i == 4:
        i+=1
    if i == 1:
        plt.ylabel("Temperature ($^\circ$C)")
    else:
        plt.tick_params('y',labelleft=False)
    distance = data.iloc[:,i+8]
    plt.xlabel(data.columns.values[i+8]+' Distance')
    plt.semilogx(distance,values,'.r',markersize=1.5,alpha=0.9)
    
    sc = np.logical_and(distance>0,distance<=30)
    co = np.logical_and(distance>30,distance<=50)
    ex = np.logical_and(distance>50,distance<=150)
    vex = np.logical_and(distance>150,distance<=1000)
    
    bins = np.array([np.median(values[sc]),np.median(values[co]),np.median(values[ex]),np.median(values[vex])])
    lower = np.array([np.percentile(values[sc],15.866),np.percentile(values[co],15.866),np.percentile(values[ex],15.866),np.percentile(values[vex],15.866)])
    upper = np.array([np.percentile(values[sc],84.134),np.percentile(values[co],84.134),np.percentile(values[ex],84.134),np.percentile(values[vex],84.134)])
    errors = [abs(bins-lower),abs(bins-upper)]
    positions = [np.median(distance[sc]),np.median(distance[co]),np.median(distance[ex]),np.median(distance[vex])]
    
    plt.errorbar(positions,bins,yerr=errors,fmt='.k',capsize=1.5)
    plt.grid(True,which='both')
    plt.xlim([1,600])
   # ax.tick_params(axis='both',labelsize=6)

#plt.suptitle("Test Axes")
plt.show()
    