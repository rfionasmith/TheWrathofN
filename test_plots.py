# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:19:17 2021

@author: rfts1
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('result_plots.pdf')
data = pd.read_csv('results.txt')
configs = [30,50,100,200,500]

for i in range(1,6):
    plt.subplot(2,3,i)
    if i < 4:
        plt.title(data.columns.values[i+4],fontsize='x-small')
        plt.semilogx(data.iloc[:,i+4],data['h'],'.r',markersize=3)
    else:
        plt.xlabel(data.columns.values[i+5],fontsize='x-small')
        plt.semilogx(data.iloc[:,i+5],data['h'],'.r',markersize=3)
    plt.grid(True,which='both')
    plt.yticks(np.arange(0,7,step=1),fontsize='xx-small')
    plt.xticks(fontsize='xx-small')
               
    for xc in configs:
        plt.axvline(x=xc,c='black',lw=1)    

    plt.xlim([1,600])
plt.subplot(2,3,6)
plt.boxplot(data['h'],sym='.',widths=1.5,positions=[1])
plt.axis('off')
plt.xlim([0,10])
plt.suptitle('Temperature Offsets')
plt.show()

'''
plt.subplot(2,3,1)
for xc in configs:
    plt.axvline(x=xc,c='coral',lw=1)
plt.semilogx(data['Total'],data['h'],'.b',markersize=3)
plt.title('Total',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(np.arange(7),fontsize='xx-small')
plt.xlim([1,600])
plt.grid(True,which='both')


plt.subplot(2,3,2)
plt.semilogx(data['EW'],data['h'],'.b',markersize=3)
plt.title('E-W',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])

plt.subplot(2,3,3)
plt.semilogx(data['NS'],data['h'],'.b',markersize=3)
plt.title('N-S',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])


plt.subplot(2,3,4)
plt.plot(data['UD'],data['h'],'.b',markersize=3)
plt.xlabel('U-D',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
#plt.xlim([1,600])
#for xc in configs:
#    plt.axvline(x=xc,c='k')

plt.subplot(2,3,5)
plt.semilogx(data['NE_SW'],data['h'],'.b',markersize=3)
plt.xlabel('NE-SW',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])

plt.subplot(2,3,6)
plt.semilogx(data['NW_SE'],data['h'],'.b',markersize=3)
plt.xlabel('NW-SE',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])

plt.suptitle('All Configs Humidity')
pp.savefig()
plt.show()


plt.subplot(2,3,1)
plt.semilogx(data['Total'],data['t'],'.r')
plt.title('Total',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,2)
plt.semilogx(data['EW'],data['t'],'.r')
plt.title('E-W',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,3)
plt.semilogx(data['NS'],data['t'],'.r')
plt.title('N-S',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,4)
plt.plot(data['UD'],data['t'],'.r')
plt.xlabel('U-D',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
#plt.xlim([1,600])
#for xc in configs:
#    plt.axvline(x=xc,c='k')

plt.subplot(2,3,5)
plt.semilogx(data['NE_SW'],data['t'],'.r')
plt.xlabel('NE-SW',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,6)
plt.semilogx(data['NW_SE'],data['t'],'.r')
plt.xlabel('NW-SE',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.suptitle('All Configs Temperature')
pp.savefig()
plt.show()


plt.subplot(2,3,1)
plt.semilogx(data['Total'],data['p'],'.g')
plt.title('Total',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,2)
plt.semilogx(data['EW'],data['p'],'.g')
plt.title('E-W',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,3)
plt.semilogx(data['NS'],data['p'],'.g')
plt.title('N-S',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,4)
plt.semilogx(data['UD'],data['p'],'.g')
plt.xlabel('U-D',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,5)
plt.semilogx(data['NE_SW'],data['p'],'.g')
plt.xlabel('NE-SW',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.subplot(2,3,6)
plt.semilogx(data['NW_SE'],data['p'],'.g')
plt.xlabel('NW-SE',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')
plt.xlim([1,600])
for xc in configs:
    plt.axvline(x=xc,c='k')

plt.suptitle('All Configs Pressure')
pp.savefig()
plt.show()
'''
pp.close()