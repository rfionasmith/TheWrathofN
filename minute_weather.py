# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

month = 'EX_Apr2020'
enu = np.loadtxt('enu_dist/EXT_AprMay_2020.txt', delimiter=',')

#month = 'EX_May2020'
#enu = np.loadtxt('enu_dist/EXT_AprMay_2020.txt', delimiter=',')

bad_ants = [6]

col_names = ['Unixtime','1','2','3','4','5','6','7','8']
t_df = pd.read_csv(month+'_t.txt',names=col_names,usecols=[0,4,5,6,7,8,9,10,11],skiprows=10,delim_whitespace=True)
t_df = t_df.drop_duplicates(subset=['Unixtime'])

p_df = pd.read_csv(month+'_p.txt',names=col_names,usecols=[0,4,5,6,7,8,9,10,11],skiprows=10,delim_whitespace=True)
p_df = p_df.drop_duplicates(subset=['Unixtime'])

col_names = ['Unixtime','1_h','2_h','3_h','4_h','5_h','6_h','7_h','8_h']  #for consistency, marge doesn't add the suffixes the second time
h_df = pd.read_csv(month+'_h.txt',names=col_names,usecols=[0,4,5,6,7,8,9,10,11],skiprows=10,delim_whitespace=True)
h_df = h_df.drop_duplicates(subset=['Unixtime'])

merge1 = pd.merge(t_df, p_df, how='outer', on='Unixtime', suffixes=('_t','_p'))
all_data = pd.merge(merge1, h_df, how='outer',on='Unixtime')
all_data = all_data.sort_values(by=['Unixtime'])    #merge tacks the extras on the end, this puts them back in time order

time = all_data['Unixtime'].values
time = np.mod(time,86400)/3600

night_mask = np.logical_and(time>4,time<16)

std_devs = pd.DataFrame(columns = ['AntA','AntB','t','p','h'])
row = 0
for idx in range(1,9):
    for jdx in range (idx+1,9):
        A = idx
        B = jdx

        antA = all_data.loc[:,[str(A) in i for i in all_data.columns]].values
        antB = all_data.loc[:,[str(B) in i for i in all_data.columns]].values

        antA = antA[night_mask]
        antB = antB[night_mask]

        temp_mask = np.logical_or(antA[:,0] > 7.5, antB[:,0] > 7.5)

        antA = antA[~temp_mask]
        antB = antB[~temp_mask]

        ### temp index 0
        tA = antA[:,0]
        tB = antB[:,0]
        ok = np.isfinite(tA) & np.isfinite(tB)
        tA = tA[ok]
        tB = tB[ok]
        
        x = np.linspace(-10,10,201)
        P = np.poly1d(np.polyfit(tA,tB,1))
        
        t_off = tB - P(tA)
        lower = np.percentile(t_off,15.866)
        upper = np.percentile(t_off,84.134)
        t_sigma = (upper-lower)/2.
        
        plt.hist2d(tA,tB,bins=201,range=[[-10.05,10.05],[-10.05,10.05]],cmap=plt.cm.plasma)
        plt.plot(x,P(x),'-y',lw=0.5)
        plt.plot(x,P(x)+lower,'--y',lw=0.5)
        plt.plot(x,P(x)+upper,'--y',lw=0.5)
        
        text = '$\sigma = %.3f ^\circ$C' % t_sigma
        plt.text(9.5,-8.5,text,c='w',horizontalalignment='right')
        plt.text(9.5,-9.5,P,c='w',horizontalalignment='right')
        
        plt.title("EX April 2020 (25m)")
        plt.xlabel("Temperature ($^\circ$C) of Ant "+str(A),fontsize='x-small')
        plt.ylabel("Temperature ($^\circ$C) of Ant "+str(B),fontsize='x-small')

        plt.show()
        
        ### pressure index 1
        pA = antA[:,1]
        pB = antB[:,1]
        ok = np.isfinite(pA) & np.isfinite(pB)
        pA = pA[ok]
        pB = pB[ok]
        
        x = np.linspace(617,633,161)
        P = np.poly1d(np.polyfit(pA,pB,1))
        
        p_off = pB - P(pA)
        lower = np.percentile(p_off,15.866)
        upper = np.percentile(p_off,84.134)
        p_sigma = (upper-lower)/2.
        
        #plt.hist2d(pA,pB,bins=161,range=[[616.95,633.05],[616.95,633.05]],cmap=#plt.cm.plasma)
        #plt.plot(x,P(x),'-y',lw=0.5)
        #plt.plot(x,P(x)+lower,'--y',lw=0.5)
        #plt.plot(x,P(x)+upper,'--y',lw=0.5)
        
        text = '$\sigma = %.3f$ mbar' % p_sigma
        #plt.text(632.3,618.2,text,c='w',horizontalalignment='right')
        #plt.text(632.3,617.4,P,c='w',horizontalalignment='right')
        
        #plt.title(month+' '+str(A)+'-'+str(B))
        #plt.xlabel("Pressure (mbar) of Ant "+str(A),fontsize='x-small')
        #plt.ylabel("Pressure (mbar) of Ant "+str(B),fontsize='x-small')

        #plt.show()
        
        ### humidity index 2
        hA = antA[:,2]
        hB = antB[:,2]
        ok = np.isfinite(hA) & np.isfinite(hB)
        hA = hA[ok]
        hB = hB[ok]
        
        x = np.linspace(0,95,96)
        P = np.poly1d(np.polyfit(hA,hB,1))
        
        h_off = hB - P(hA)
        lower = np.percentile(h_off,15.866)
        upper = np.percentile(h_off,84.134)
        h_sigma = (upper-lower)/2.
        
        #plt.hist2d(hA,hB,bins=96,range=[[-0.5,95.5],[-0.5,95.5]],cmap=plt.cm.plasma)
        #plt.plot(x,P(x),'-y',lw=0.5)
        #plt.plot(x,P(x)+lower,'--y',lw=0.5)
        #plt.plot(x,P(x)+upper,'--y',lw=0.5)
        
        text = ('$\sigma = %.3f$' % h_sigma)+' %'
        #plt.text(94.5,5.5,text,c='w',horizontalalignment='right')
        #plt.text(94.5,1.5,P,c='w',horizontalalignment='right')
        
        #plt.title(month+' '+str(A)+'-'+str(B))
        #plt.xlabel("Humidity (%) of Ant "+str(A),fontsize='x-small')
        #plt.ylabel("Humidity (%) of Ant "+str(B),fontsize='x-small')

        #plt.show()
                
        std_devs.loc[row] = [A,B,t_sigma,p_sigma,h_sigma]
        row+=1
print(std_devs)

########################### getting the baselines

#enu = np.loadtxt('enu_dist/SUB_NovDec_2020.txt', delimiter=',')
#baseline = np.loadtxt('baselines/SUB_NovDec_2020.txt', delimiter=',')
#baseline = np.sort(baseline.view('f8,f8,f8,f8,f8'),order=['f0','f1'],axis=0).view(float)

#plt.plot(baseline[:,4],std_devs['h'],'.r')
#plt.title('Width vs. Baseline (Temp)')
#plt.show()
#bad_ants = [5,6]
total_dist = np.sqrt(enu[:,2]**2+enu[:,3]**2+enu[:,4]**2)
ew = abs(enu[:,2])
ns = abs(enu[:,3])
ud = abs(enu[:,4])
ne_sw = abs(enu[:,3]+enu[:,2])/np.sqrt(2)
nw_se = abs(enu[:,3]-enu[:,2])/np.sqrt(2)
dist = pd.DataFrame({'AntA':enu[:,0],'AntB':enu[:,1],'Total':total_dist,
                     'EW':ew,'NS':ns,'UD':ud,'NE_SW':ne_sw,'NW_SE':nw_se})

'''
### Shenanigans for when ants are missing positions
flag1 = np.logical_or(std_devs['AntA']==6,std_devs['AntB']==6)
std_devs = std_devs[~flag1]
std_devs = std_devs.reset_index(drop=True)
dist = dist.reset_index(drop=True)
flag2 = np.logical_or(std_devs['AntA']==6,std_devs['AntB']==6)
std_devs = std_devs[~flag2]
dist = dist[~flag2]    

for ant in bad_ants:
    flag = np.logical_or(std_devs['AntA']==ant,std_devs['AntB']==ant)
    enu = enu[~flag]
    dist = dist[~flag]
    std_devs = std_devs[~flag]

### e = 2, n = 3, u = 4
plt.subplot(2,3,1)
plt.plot(dist['Total'],std_devs['t'],'.r')
plt.title('Total',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,2)
plt.plot(dist['EW'],std_devs['t'],'.r')
plt.title('E-W',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,3)
plt.plot(dist['NS'],std_devs['t'],'.r')
plt.title('N-S',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,4)
plt.plot(dist['UD'],std_devs['t'],'.r')
plt.xlabel('U-D',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,5)
plt.plot(dist['NE_SW'],std_devs['t'],'.r')
plt.xlabel('NE-SW',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,6)
plt.plot(dist['NW_SE'],std_devs['t'],'.r')
plt.xlabel('NW-SE',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.suptitle(month+' Temperature')
plt.show()

### humidity
plt.subplot(2,3,1)
plt.plot(dist['Total'],std_devs['h'],'.b')
plt.title('Total',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,2)
plt.plot(dist['EW'],std_devs['h'],'.b')
plt.title('E-W',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,3)
plt.plot(dist['NS'],std_devs['h'],'.b')
plt.title('N-S',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,4)
plt.plot(dist['UD'],std_devs['h'],'.b')
plt.xlabel('U-D',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,5)
plt.plot(dist['NE_SW'],std_devs['h'],'.b')
plt.xlabel('NE-SW',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,6)
plt.plot(dist['NW_SE'],std_devs['h'],'.b')
plt.xlabel('NW-SE',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.suptitle(month+' Humidity')
plt.show()


### pressure
plt.subplot(2,3,1)
plt.plot(dist['Total'],std_devs['p'],'.g')
plt.title('Total',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,2)
plt.plot(dist['EW'],std_devs['p'],'.g')
plt.title('E-W',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,3)
plt.plot(dist['NS'],std_devs['p'],'.g')
plt.title('N-S',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,4)
plt.plot(dist['UD'],std_devs['p'],'.g')
plt.xlabel('U-D',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,5)
plt.plot(dist['NE_SW'],std_devs['p'],'.g')
plt.xlabel('NE-SW',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,6)
plt.plot(dist['NW_SE'],std_devs['p'],'.g')
plt.xlabel('NW-SE',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.suptitle(month+' Pressure')
plt.show()

### just to collect all the data in a file to see what's up, there is definitely a better way to do this
### see result_plots.pdf for the actual plots/file
old_record = pd.read_csv('results.txt')
new_results = pd.merge(std_devs,dist, on=['AntA','AntB'])
new_record = pd.concat([old_record,new_results])
new_record.to_csv('results.txt', index=False)


plt.subplot(2,3,1)
plt.plot(new_record['Total'],new_record['h'],'.b')
plt.title('Total',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,2)
plt.plot(new_record['EW'],new_record['h'],'.b')
plt.title('E-W',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,3)
plt.plot(new_record['NS'],new_record['h'],'.b')
plt.title('N-S',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,4)
plt.plot(new_record['UD'],new_record['h'],'.b')
plt.xlabel('U-D',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,5)
plt.plot(new_record['NE_SW'],new_record['h'],'.b')
plt.xlabel('NE-SW',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.subplot(2,3,6)
plt.plot(new_record['NW_SE'],new_record['h'],'.b')
plt.xlabel('NW-SE',fontsize='small')
plt.xticks(fontsize='xx-small')
plt.yticks(fontsize='xx-small')

plt.suptitle('All Configs Humidity')
plt.show()

#'''