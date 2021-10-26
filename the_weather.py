
import numpy as np
import matplotlib.pyplot as plt


def get_median_offsets(f_name, ant_list, weather_type):
    data = np.loadtxt(f_name,usecols=(3,4,5,6,7,8,9,10,11),skiprows=8)
    time = np.loadtxt(f_name,usecols=0,skiprows=8)
    time = np.mod(time,86400)/3600
    
    night_mask = np.logical_and(time>6,time<16)
    good_ants = np.array(ant_list)
    good_data = data[:,good_ants]    
    n = len(ant_list)
    
    
    index_arr = np.sort(np.append(np.where(night_mask[1:] != night_mask[:-1])[0], [-1,night_mask.size]))
    if night_mask[0]:
        start = index_arr[0:-1:2]+1
        end = index_arr[1::2]+1
    else:
        start = index_arr[1:-1:2]+1
        end = index_arr[2::2]+1
        

    day=1    
    for i,j in zip(start,end):
        temp_data = good_data[i:j]
        temp_time = time[i:j]
        '''
        for ant in range(0,n):
            plt.plot(temp_time,temp_data[:,ant],'.',label=str(good_ants[ant]),lw=0.5)
        plt.title("day "+str(day))
        plt.legend(fontsize='x-small')
        plt.show()
        '''
        median=np.median(temp_data,axis=1)
        offsets = temp_data-median[:,None]
        
        off_medians = np.percentile(offsets,50.,axis=0)
        percent_25 = np.percentile(offsets,25.,axis=0)
        percent_75 = np.percentile(offsets,75.,axis=0)
        
        if day == 1:
            final_medians = off_medians
            final_25 = percent_25
            final_75 = percent_75
        else:
            final_medians = np.vstack([final_medians,off_medians])
            final_25 = np.vstack([final_25,percent_25])
            final_75 = np.vstack([final_75,percent_75])
            
        day+=1
        
    period = range(1,day)
    for ant in range(0,n):
        #percentile gives actual value, but errorbar needs positive difference from datapoint
        lower = final_medians[:,ant]-final_25[:,ant]
        upper = final_75[:,ant]-final_medians[:,ant]
        errors = [lower,upper]
        
        #plt.plot(period,final_medians[:,ant],label=str(good_ants[ant]),lw=1)
        plt.errorbar(period,final_medians[:,ant],yerr=errors,label=str(good_ants[ant]),fmt='.-',lw=1,alpha=.8,capsize=1.5)
        
        #plt.plot(period,final_25[:,ant],lw=1,alpha=0.5)
        #plt.plot(period,final_75[:,ant],lw=1,alpha=0.5)
    plt.title(f_name)
    plt.xlabel("Days")
    
    #just to get y-axis labels
    if weather_type == "H":
        plt.ylabel("% Humidity")
    if weather_type == "P":
        plt.ylabel("millbars")
    if weather_type == "T":
        plt.ylabel("Degrees Celsius")
    plt.legend(fontsize='x-small')
    plt.show()
    #print(f_name)
    #print(final_medians)
###############################################


ants = [1,2,3,4,5,7,8]      #6's humidity is always off

#nov 1-30 2020
get_median_offsets("SC_Nov2020_h.txt",ants,"H")
get_median_offsets("SC_Nov2020_p.txt",ants,"P")
get_median_offsets("SC_Nov2020_t.txt",ants,"T")

#dec 1-31 2020
get_median_offsets("SC_Dec2020_h.txt",ants,"H")
get_median_offsets("SC_Dec2020_p.txt",ants,"P")
get_median_offsets("SC_Dec2020_t.txt",ants,"T")

ants = [1,2,4,5,7,8]
#apr 1-30 2021,    3 in the hangar
get_median_offsets("CO_Apr2021_h.txt",ants,"H")
get_median_offsets("CO_Apr2021_p.txt",ants,"P")
get_median_offsets("CO_Apr2021_t.txt",ants,"T")

#jul 1-31 2021,    7 in the hangar
ants = [1,2,3,4,5,8]
get_median_offsets("CO_Jul2021_h.txt",ants,"H")
get_median_offsets("CO_Jul2021_p.txt",ants,"P")
get_median_offsets("CO_Jul2021_t.txt",ants,"T")

ants = [1,2,5,7,8]      #ant 3 was put in the hangar mid september, 4 is crazy

#sept 1-30 2021
get_median_offsets("SC_Sep2021_h.txt",ants,"H")
get_median_offsets("SC_Sep2021_p.txt",ants,"P")
get_median_offsets("SC_Sep2021_t.txt",ants,"T")

#oct 1-7 2021
#get_median_offsets("humidity_1wk.txt",ants,"H")
#get_median_offsets("pressure_1wk.txt",ants,"P")
#get_median_offsets("temp_1wk.txt",ants,"T")


