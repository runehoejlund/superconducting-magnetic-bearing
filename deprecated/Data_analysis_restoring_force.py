# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:55:21 2021

Data analysis of restoring force experiments

@author: joachim
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.ndimage.filters import uniform_filter1d
from datetime import datetime


#Importing of data
data_E1 = pd.read_csv(r'C:\Users\joach\OneDrive - Danmarks Tekniske Universitet\Wordstuff-copy\Uddannelse\Univeristetet\Kandidat\2 semester\X-tech\3_week_course+Article\Experiment_only_inner\Restoring force\Experimental_data\Experiment_1_restoring_force.csv',delimiter=',')

df_E1 = pd.DataFrame(data_E1)#, columns= ['t','x-force','y-force','z-force']

#print(df_E1)

Time=np.zeros(len(df_E1['Time']))
for i in range(len(df_E1['Time'])):
    #time of day first spliting of date, then seperating hours,min and seconds
    tod=df_E1['Time'][i].split(' ')[1].split(':')
    h=float(tod[0])
    m=float(tod[1])
    s=float(tod[2])
    #print(h,m,s)
    Time[i]=h*3600+m*60+s
Time-=Time[0]

Forces=df_E1.values[:,1:].astype('float64')

print(Forces)
#y
#plt.plot(Time[7800:23200],Forces[7800:23200:,2])
#plt.plot(Time[9400:9600],Forces[9400:9600:,2])
#plt.show()
#x
#plt.plot(Time[31000:42000],Forces[31000:42000,0])
#plt.plot(Time[9400:9600],Forces[9400:9600:,2])
plt.plot(Forces)
plt.show()


#%% Sectioning each of the steps of the stairs

'''
My plan for sectioning the data is by applying a moving average
and using that to find when the slope is flat. Then I willl sort
out the adjustment period by setting a limit on how long the 
plateau is.
After finding the sections of each of the stairs I can trace that
back to the index and then do the data analysis from there.
Make sure to test the sorting by ploting them together

'''
def Sectioning(Force,Time,plot_option):
    
    #smoothing of data
    Test_Force1_c1 = uniform_filter1d(Force, size=15)
    
    #finding each of the sharp slopes
    m = 5 #amount of points the slope is found over
    step_limit = 0.05
    
    Index_list_s = [] #start points for each sudden peak
    for i in range(len(Test_Force1_c1)-m):
        if Test_Force1_c1[i+m]-Test_Force1_c1[i]>step_limit:
            Index_list_s.append(i)
    Index_list_s_c1=[]
    for i in range(len(Index_list_s)):
        if Index_list_s[i]-Index_list_s[i-1]!=1:
            Index_list_s_c1.append(int(Index_list_s[i]))
            
    Index_list_e = [] #end points for each sudden peak
    for i in range(len(Test_Force1_c1)-m):
        if Test_Force1_c1[i]-Test_Force1_c1[i-m]>step_limit:
            Index_list_e.append(i)
    Index_list_e_c1=[]
    for i in range(len(Index_list_e)-1):
        if Index_list_e[i+1]-Index_list_e[i]!=1:
            Index_list_e_c1.append(int(Index_list_e[i]))
    Index_list_e_c1.append(Index_list_e[len(Index_list_e)-1])
    
    
    if len(Index_list_e_c1)!=len(Index_list_s_c1):
        print('ERROR: not equal amount of starts and ends when indexing')
    
    #Now  we will sort each of the peaks so they come in pairs
    L = 225 #minimum flat region between peaks
    p = 0.3 #procentage of the data it is averaged over
    Index_list_flat_s_c2=[]
    Index_list_flat_e_c2=[]
    for i in range(len(Index_list_e_c1)-1):
        if Index_list_s_c1[i+1]-Index_list_e_c1[i]>L:
            Index_list_flat_s_c2.append(Index_list_e_c1[i])
            Index_list_flat_e_c2.append(Index_list_s_c1[i+1])
    #print(len(Index_list_flat_s_c2),len(Index_list_flat_s_c2))
    if plot_option!='No':
        plt.figure()
        Out=Force
        plt.plot(Time,Out,'k')
    Force_avg=np.zeros(len(Index_list_flat_s_c2))
    for i in range(len(Index_list_flat_s_c2)):
            Index_End=math.floor((Index_list_flat_e_c2[i]-Index_list_flat_s_c2[i])*p+Index_list_flat_s_c2[i])
            Force_avg[i] = np.average(Force[Index_list_flat_s_c2[i]:Index_End])
            #plt.plot(Time[Index_list_flat_s_c2[i]:Index_list_flat_e_c2[i]],Force[Index_list_flat_s_c2[i]:Index_list_flat_e_c2[i]],'m')
            if plot_option!='No':
                plt.plot(Time[Index_list_flat_s_c2[i]:Index_End],Force[Index_list_flat_s_c2[i]:Index_End],'m')
    if plot_option!='No':
        plt.show()
    
    #print(Index_list_s_c1,Index_list_e_c1)
    #for i in range(len(Index_list_s_c1)):
    #    if Index_list_e_c1[i]-Index_list_s_c1[i]>15:
    #        plt.plot(Time[Index_list_s_c1[i]:Index_list_e_c1[i]],Force[Index_list_s_c1[i]:Index_list_e_c1[i]],'r')
    #plt.xlim(83,110)
    #plt.ylim(1.2,6.2)
    
    dist=0.05 #mm
    displacement=np.arange(0,(len(Force_avg)-0.5)*dist,dist)
    if plot_option!='No':
        plt.plot(displacement,Force_avg)
    #~25N/mm
    return(displacement,Force_avg)

#setting a test set

#x-data
Test_Force_x = Forces[31000:42000,0]
Test_Time_x  = Time[31000:42000]
 
#y-data
Test_Force_y = Forces[8400:21660,1]
Test_Time_y  = Time[8400:21660]

displacement_x,Force_avg_x=Sectioning(Test_Force_x,Test_Time_x,'N2o')
displacement_y,Force_avg_y=Sectioning(Test_Force_y,Test_Time_x,'N2o')

Ax=np.vstack([displacement_x, np.ones(len(displacement_x))]).T

mx, cx = np.linalg.lstsq(Ax, Force_avg_x, rcond=None)[0]

#Ay=np.vstack([displacement_y, np.ones(len(displacement_y))]).T

#my, cy = np.linalg.lstsq(Ay, Force_avg_y, rcond=None)[0]
#%%
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
fontticksize = 12
#pathcollection
plt.figure()
plt.title('Radial Stiffness of Radial bearing',fontdict=font)

plt.plot(displacement_x,Force_avg_x,label='x-Force')
plt.plot(displacement_y,Force_avg_y,label='y-Force')
#plt.plot(displacement_x, mx*displacement_x + cx, 'r', label='x-Fitted line')
plt.grid()
plt.legend()
#plt.xlim(0,5.5)
#plt.ylim(2500,5000)
plt.tick_params(labelsize=fontticksize)
plt.xlabel(r'displacement [mm]', fontdict=font)
plt.ylabel(r'Force [N]', fontdict=font)
#plt.plot(kappa_m[1:-2],diffrho[1:-2],'.')
plt.show()

plt.figure()

plt.show()
