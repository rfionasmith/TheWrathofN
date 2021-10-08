
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
'''
plt.plot(t['unixtime'],t['ant1'],lw=.5)
plt.plot(t['unixtime'],t['ant2'],lw=.5)
#plt.plot(t['unixtime'],t['ant3'],lw=.5)    #in hangar
plt.plot(t['unixtime'],t['ant4'],lw=.5)
plt.plot(t['unixtime'],t['ant5'],lw=.5)
plt.plot(t['unixtime'],t['ant6'],lw=.5)
plt.plot(t['unixtime'],t['ant7'],lw=.5)
plt.plot(t['unixtime'],t['ant8'],lw=.5)
plt.show()
'''

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
'''
plt.plot(t_offsets[0],t_offsets[1],lw=.5)
plt.plot(t_offsets[0],t_offsets[2],lw=.5)
#plt.plot(t_offsets[0],t_offsets[3],lw=.5)      #in hangar
plt.plot(t_offsets[0],t_offsets[4],lw=.5)
plt.plot(t_offsets[0],t_offsets[5],lw=.5)
#plt.plot(t_offsets[0],t_offsets[6],lw=.5)      #referencing itself
plt.plot(t_offsets[0],t_offsets[7],lw=.5)
plt.plot(t_offsets[0],t_offsets[8],lw=.5)
plt.show()
'''
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

