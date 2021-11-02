# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:18:35 2021

@author: rfts1

"""
import numpy as np
import matplotlib.pyplot as plt

f_name = "SC_Nov2020_p.txt"

data = np.loadtxt(f_name,usecols=(3,4,5,6,7,8,9,10,11),skiprows=8)
time = np.loadtxt(f_name,usecols=0,skiprows=8)
time = np.mod(time,86400)/3600

good_ants = np.array([1,2,3,4,5,6,7,8])
n = len(good_ants)
data = data[:,good_ants]

for hr in range(0,24):
    #creates a mask for each hour, so I can work with all the 00:00-00:59 etc. data at once
    print(hr,hr+1)
    hour_mask = np.logical_and(time>=hr,time<hr+1)
    index_arr = np.sort(np.append(np.where(hour_mask[1:] != hour_mask[:-1])[0], [-1,hour_mask.size]))
    if hour_mask[0]:
        start = index_arr[0:-1:2]+1
        end = index_arr[1::2]+1
    else:
        start = index_arr[1:-1:2]+1
        end = index_arr[2::2]+1 
        
    for i,j in zip(start,end): 
        #steps through the masked hour for each day in the set
        median = np.median(data[i:j],axis=1)    #median values of weather data across array per min
        
        #print(hr,len(median))
        #if len(median) != 60:
        #    continue
        offsets = data[i:j]-median[:,None]      #offset of weather value from median across array per min
        
        #next takes the median value of the offsets throughout the hour for each antenna and puts in a separate array
        #i.e. day 1, hr 1, 60 offset values for 1 ant --> 1 value for 1 ant. hr_offset will be: # of days in set X # ants
        if i == start[0]:
            hr_offset = np.median(offsets,axis=0)
        else:
            hr_offset = np.vstack((hr_offset,np.median(offsets,axis=0))) ###need to figure out how to deal with missing minutes

        #plt.plot(time[i:j],data[i:j,1],'.')
        #plt.plot(time[i:j],offsets[:,0],'.')
        #plt.title(hr)
    #plt.show()
    #print(hr_offset)
    
    #next gets the median value of all the offsets for a specific hr across the dataset per antenna
    #i.e. 1 month hr_offset has 30 values for 1 ant --> 1 value/ant. final_result will be: 24 X # ants
    if hr == 0:    
        final_result = np.median(hr_offset,axis=0)
    else:
        final_result = np.vstack((final_result,np.median(hr_offset,axis=0)))
    print('-----') 
print(final_result)
for i in range(0,n):
    plt.plot(final_result[:,i],label=good_ants[i])
plt.xlabel('UTC hour')
plt.ylabel('Offset each hr over the month')
plt.legend(fontsize='x-small')
plt.title(f_name)
plt.show()
    