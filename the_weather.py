
import numpy as np
import matplotlib.pyplot as plt


def get_median_offsets(f_name, ant_list, weather_type):
    original_data = np.loadtxt(f_name,usecols=(3,4,5,6,7,8,9,10,11),skiprows=8)
    original_time = np.loadtxt(f_name,usecols=0,skiprows=8)
    original_time = np.mod(original_time,86400)/3600
    night_mask = np.logical_and(original_time>6,original_time<16)
    good_ants = np.array(ant_list)
    n = len(ant_list)

    night_time = original_time[night_mask]
    night_data = original_data[night_mask]
    night_data = night_data[:,good_ants]

    day = 0
    i = 0
    while i in range(0,len(night_time)):
        if i == 0:
            temp_time = night_time[0]
            temp_data = night_data[0]
            i+=1
            continue
        if night_time[i] < night_time[i-1]:
            day+=1      
            ### get medians and offsets
            median = np.median(temp_data,axis=1)
            offsets = temp_data-median[:,None]
                
            off_medians = np.median(offsets,axis=0) 
            if day == 1:
                final_medians = off_medians
            else:
                final_medians = np.vstack([final_medians,off_medians])     
                
            #PLOT
            '''
            for j in range(0,n):
                plt.plot(temp_time,temp_data[:,j],'.',label=str(good_ants[j]),lw=0.5)
            plt.title("Day "+str(day))
            plt.legend(fontsize='x-small')
            plt.show()
             
            for k in range(0,n):
                plt.plot(temp_time,offsets[:,k],'.',label=str(good_ants[k]),lw=0.5)
            plt.title("Day "+str(day)+" Offsets")
            plt.legend(fontsize='x-small')
            plt.show()
            '''
            #RESET FOR NEXT DAY
            temp_time = night_time[i] 
            temp_data = night_data[i]
            i+=1
            next
        temp_time = np.append(temp_time, night_time[i])
        temp_data = np.vstack([temp_data,night_data[i]])
        i+=1
    
    ###redo everything for the last day in the file, should probably make another function for this
    day+=1  

    median = np.median(temp_data,axis=1)
    offsets = temp_data-median[:,None]

    off_medians = np.median(offsets,axis=0)
    final_medians = np.vstack([final_medians,off_medians])
    '''
    #PLOTS
    for j in range(0,n):
        plt.plot(temp_time,temp_data[:,j],'.',label=str(good_ants[j]),lw=0.5)
    plt.title("Day"+str(day))
    plt.legend(fontsize='x-small')
    plt.show()

    for k in range(0,n):
        plt.plot(temp_time,offsets[:,k],'.',label=str(good_ants[k]),lw=0.5)
    plt.title("Day "+str(day)+" Offsets")
    plt.legend(fontsize='x-small')
    plt.show()
    ''' 

    period = range(1,day+1)     #just to label the x-axis for now, will need to get actual dates later
    for m in range(0,n):
        plt.plot(period,final_medians[:,m],label=str(good_ants[m]),lw=1)
    plt.title(weather_type+" Median Offsets Oct 1-7 UTC")
    plt.xlabel("Days")
    if weather_type == "H":
        plt.ylabel("% Humidity")
    if weather_type == "P":
        plt.ylabel("millbars")
    if weather_type == "T":
        plt.ylabel("Degrees Celsius")
    plt.legend(fontsize='x-small')
    plt.show()
    print(f_name)
    print(final_medians)
###############################################


ants = [1,2,5,7,8]

#sept 1-30 2021
#get_median_offsets("humidity_1mo.txt",ants,"H")
#get_median_offsets("pressure_1mo.txt",ants,"P")
#get_median_offsets("temp_1mo.txt",ants,"T")

#oct 1-7 2021
get_median_offsets("humidity_1wk.txt",ants,"H")
get_median_offsets("pressure_1wk.txt",ants,"P")
get_median_offsets("temp_1wk.txt",ants,"T")
