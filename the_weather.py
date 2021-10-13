
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#np array probably better for getting the median, etc...
h_data = np.loadtxt("humidity_24hr.txt", usecols=(3,4,5,6,7,8,9,10),skiprows=8)
h_time = np.loadtxt("humidity_24hr.txt", usecols=0,skiprows=8)

h_median = np.median(h_data,axis=1)
h_offsets_1 = h_data[:,0]-h_median
h_offsets_2 = h_data[:,1]-h_median
h_offsets_3 = h_data[:,2]-h_median
h_offsets_4 = h_data[:,3]-h_median
h_offsets_5 = h_data[:,4]-h_median
h_offsets_6 = h_data[:,5]-h_median
h_offsets_7 = h_data[:,6]-h_median
h_offsets_8 = h_data[:,7]-h_median


for i in range(0,8):
    plt.plot(h_time,h_data[:,i],label=str(i+1),lw=0.5)
plt.plot(h_time,h_median,label='median',lw=.5,alpha=.5,color='blue')
plt.legend(fontsize='x-small')
plt.xlim(1633406760,1633449840)          #start time ~6pm, unixtime of ~6am
plt.ylim(0,120)
plt.title("Humidity")
plt.show()

plt.plot(h_time,h_median,label='median',alpha=.5)
plt.plot(h_time,h_offsets_1,label='ant1',lw=0.5)
plt.plot(h_time,h_offsets_2,label='ant2',lw=0.5)
#plt.plot(h_time,h_offsets_3,label='ant3',lw=0.5)
#plt.plot(h_time,h_offsets_4,label='ant4',lw=0.5)
plt.plot(h_time,h_offsets_5,label='ant5',lw=0.5)
plt.plot(h_time,h_offsets_6,label='ant6',lw=0.5)
plt.plot(h_time,h_offsets_7,label='ant7',lw=0.5)
plt.plot(h_time,h_offsets_8,label='ant8',lw=0.5)
plt.legend(fontsize='xx-small')
plt.xlim(1633406760,1633449840)          #start time ~6pm, unixtime of ~6am
plt.ylim(-20,20)
plt.title("Humidity Offsets")
plt.show()


t_data = np.loadtxt("temp_24hr.txt", usecols=(3,4,5,6,7,8,9,10),skiprows=8)
t_time = np.loadtxt("temp_24hr.txt", usecols=0,skiprows=8)

t_median = np.median(t_data,axis=1)
t_offsets_1 = t_data[:,0]-t_median
t_offsets_2 = t_data[:,1]-t_median
t_offsets_3 = t_data[:,2]-t_median
t_offsets_4 = t_data[:,3]-t_median
t_offsets_5 = t_data[:,4]-t_median
t_offsets_6 = t_data[:,5]-t_median
t_offsets_7 = t_data[:,6]-t_median
t_offsets_8 = t_data[:,7]-t_median

for i in range(0,8):
    plt.plot(t_time,t_data[:,i],label=str(i+1),lw=0.5)
plt.plot(t_time,t_median,label='median',lw=.5,alpha=.5,color='blue')
plt.legend(fontsize='xx-small')
plt.xlim(1633406760,1633449840)          #start time ~6pm, unixtime of ~6am
#plt.ylim(0,120)
plt.title("Temperature")
plt.show()

plt.plot(t_time,t_median,label='median',alpha=.5)
plt.plot(t_time,t_offsets_1,label='ant1',lw=0.5)
plt.plot(t_time,t_offsets_2,label='ant2',lw=0.5)
#plt.plot(t_time,t_offsets_3,label='ant3',lw=0.5)
#plt.plot(t_time,t_offsets_4,label='ant4',lw=0.5)
plt.plot(t_time,t_offsets_5,label='ant5',lw=0.5)
plt.plot(t_time,t_offsets_6,label='ant6',lw=0.5)
plt.plot(t_time,t_offsets_7,label='ant7',lw=0.5)
plt.plot(t_time,t_offsets_8,label='ant8',lw=0.5)
plt.legend(fontsize='xx-small')
plt.xlim(1633406760,1633449840)          #start time ~6pm, unixtime of ~6am
#plt.ylim(-20,20)
plt.title("Temperature Offsets")
plt.show()


p_data = np.loadtxt("pressure_24hr.txt", usecols=(3,4,5,6,7,8,9,10),skiprows=8)
p_time = np.loadtxt("pressure_24hr.txt", usecols=0,skiprows=8)

p_median = np.median(p_data,axis=1)
p_offsets_1 = p_data[:,0]-p_median
p_offsets_2 = p_data[:,1]-p_median
p_offsets_3 = p_data[:,2]-p_median
p_offsets_4 = p_data[:,3]-p_median
p_offsets_5 = p_data[:,4]-p_median
p_offsets_6 = p_data[:,5]-p_median
p_offsets_7 = p_data[:,6]-p_median
p_offsets_8 = p_data[:,7]-p_median

for i in range(0,8):
    plt.plot(p_time,p_data[:,i],label=str(i+1),lw=0.5)
plt.plot(p_time,p_median,label='median',lw=.5,alpha=.5,color='blue')
plt.legend(fontsize='xx-small',loc='lower right')
plt.xlim(1633406760,1633449840)          #start time ~6pm, unixtime of ~6am
plt.ylim(615,630)
plt.title("Pressure")
plt.show()

#plt.plot(p_time,p_median,label='median',alpha=.5)
plt.plot(p_time,p_offsets_1,label='ant1',lw=0.5)
plt.plot(p_time,p_offsets_2,label='ant2',lw=0.5)
#plt.plot(p_time,p_offsets_3,label='ant3',lw=0.5)
#plt.plot(p_time,p_offsets_4,label='ant4',lw=0.5)
plt.plot(p_time,p_offsets_5,label='ant5',lw=0.5)
plt.plot(p_time,p_offsets_6,label='ant6',lw=0.5)
plt.plot(p_time,p_offsets_7,label='ant7',lw=0.5)
plt.plot(p_time,p_offsets_8,label='ant8',lw=0.5)
plt.legend(fontsize='xx-small',loc='center right')
plt.xlim(1633406760,1633449840)          #start time ~6pm, unixtime of ~6am
#plt.ylim(20,20)
plt.title("Pressure Offsets")
plt.show()


'''
#get the data
cols = ['unixtime','UTC','HST','ant1','ant2','ant3','ant4','ant5','ant6','ant7','ant8']
t = pd.read_table("temp_24hr.txt",sep='\s+',skiprows=10,names=cols)
p = pd.read_table("pressure_24hr.txt",sep='\s+',skiprows=10,names=cols)
h = pd.read_table("humidity_24hr.txt",sep='\s+',skiprows=10,names=cols)

#night time only: 18:04-6:04(1633449840)
t = t[t['unixtime'] <= 1633449840]  
p = p[p['unixtime'] <= 1633449840]
h = h[h['unixtime'] <= 1633449840]

ref = 'ant6'

#just working with temperature for now

plt.plot(t['unixtime'],t['ant1'],lw=.5)
plt.plot(t['unixtime'],t['ant2'],lw=.5)
#plt.plot(t['unixtime'],t['ant3'],lw=.5)    #in hangar
plt.plot(t['unixtime'],t['ant4'],lw=.5)
plt.plot(t['unixtime'],t['ant5'],lw=.5)
plt.plot(t['unixtime'],t['ant6'],lw=.5)
plt.plot(t['unixtime'],t['ant7'],lw=.5)
plt.plot(t['unixtime'],t['ant8'],lw=.5)
plt.show()


t_offsets = np.array([t['unixtime'].values,
                   (t['ant1']-t[ref]).values,
                   (t['ant2']-t[ref]).values,
                   (t['ant3']-t[ref]).values,
                   (t['ant4']-t[ref]).values,
                   (t['ant5']-t[ref]).values,
                   (t['ant6']-t[ref]).values,
                   (t['ant7']-t[ref]).values,
                   (t['ant8']-t[ref]).values])

#print(len(diff1.values))
#print(t['unixtime'])

plt.plot(t_offsets[0],t_offsets[1],lw=.5)
plt.plot(t_offsets[0],t_offsets[2],lw=.5)
#plt.plot(t_offsets[0],t_offsets[3],lw=.5)      #in hangar
plt.plot(t_offsets[0],t_offsets[4],lw=.5)
plt.plot(t_offsets[0],t_offsets[5],lw=.5)
#plt.plot(t_offsets[0],t_offsets[6],lw=.5)      #referencing itself
plt.plot(t_offsets[0],t_offsets[7],lw=.5)
plt.plot(t_offsets[0],t_offsets[8],lw=.5)
plt.show()

#print(t_offsets)


### pressure
plt.plot(p['unixtime'],p['ant1'],lw=.5)
plt.plot(p['unixtime'],p['ant2'],lw=.5)
plt.plot(p['unixtime'],p['ant3'],lw=.5)
#plt.plot(p['unixtime'],p['ant4'],lw=.5)
plt.plot(p['unixtime'],p['ant5'],lw=.5)
plt.plot(p['unixtime'],p['ant6'],lw=.5)
plt.plot(p['unixtime'],p['ant7'],lw=.5)
plt.plot(p['unixtime'],p['ant8'],lw=.5)
plt.show()

p_offsets = np.array([p['unixtime'].values,
                   (p['ant1']-p[ref]).values,
                   (p['ant2']-p[ref]).values,
                   (p['ant3']-p[ref]).values,
                   (p['ant4']-p[ref]).values,
                   (p['ant5']-p[ref]).values,
                   (p['ant6']-p[ref]).values,
                   (p['ant7']-p[ref]).values,
                   (p['ant8']-p[ref]).values])

plt.plot(p_offsets[0],p_offsets[1],lw=.5)
plt.plot(p_offsets[0],p_offsets[2],lw=.5)
#plt.plot(p_offsets[0],p_offsets[3],lw=.5)      #in hangar
#plt.plot(p_offsets[0],p_offsets[4],lw=.5)      #broke
plt.plot(p_offsets[0],p_offsets[5],lw=.5)
#plt.plot(p_offsets[0],p_offsets[6],lw=.5)      #referencing itself
plt.plot(p_offsets[0],p_offsets[7],lw=.5)
plt.plot(p_offsets[0],p_offsets[8],lw=.5)
plt.show()

### humidity
plt.plot(h['unixtime'],h['ant1'],lw=.5)
plt.plot(h['unixtime'],h['ant2'],lw=.5)
#plt.plot(h['unixtime'],h['ant3'],lw=.5)       #in hangar
#plt.plot(h['unixtime'],h['ant4'],lw=.5)       #broke
plt.plot(h['unixtime'],h['ant5'],lw=.5)
plt.plot(h['unixtime'],h['ant6'],lw=.5)
plt.plot(h['unixtime'],h['ant7'],lw=.5)
plt.plot(h['unixtime'],h['ant8'],lw=.5)
plt.show()

h_offsets = np.array([h['unixtime'].values,
                   (h['ant1']-h[ref]).values,
                   (h['ant2']-h[ref]).values,
                   (h['ant3']-h[ref]).values,
                   (h['ant4']-h[ref]).values,
                   (h['ant5']-h[ref]).values,
                   (h['ant6']-h[ref]).values,
                   (h['ant7']-h[ref]).values,
                   (h['ant8']-h[ref]).values])

plt.plot(h_offsets[0],h_offsets[1],lw=.5)
plt.plot(h_offsets[0],h_offsets[2],lw=.5)
#plt.plot(h_offsets[0],h_offsets[3],lw=.5)      #in hangar
#plt.plot(h_offsets[0],h_offsets[4],lw=.5)      #broke
plt.plot(h_offsets[0],h_offsets[5],lw=.5)
#plt.plot(h_offsets[0],h_offsets[6],lw=.5)      #referencing itself
plt.plot(h_offsets[0],h_offsets[7],lw=.5)
plt.plot(h_offsets[0],h_offsets[8],lw=.5)
plt.show()

'''