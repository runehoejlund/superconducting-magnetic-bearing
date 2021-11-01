# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 21:17:26 2021

@author: joachim
"""

#Damping experiment
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
fontticksize = 12

#Data set
#Data_set=1

#Importing data
#time is in the first collumb
dfRadial = pd.read_csv (r'./Radial_data/Damping_experiment_radial.csv',skiprows=1)
dfAxial = pd.read_csv (r'./Axial_data/Damping_axial_succes_slightly_off_center_2_part1.csv',skiprows=1)


#Making numpy array
rad=dfRadial.to_numpy()
#mistake
axe=dfAxial.to_numpy()

#Cleaning data
d1=axe[:,1].astype(np.float)
print(d1)
#print(np.shape(q))
#print(q)
#removing nan data
d1 = d1[~np.isnan(d1)]
#Adding time collumb in the start
#d1=np.vstack((q[:len(d1),0],d1))

#Plotting all of the data
plt.plot(d1)

#Finding the peaks of the data
#calculating mean and standard deviation
mean1=np.mean(d1)
std1=np.std(d1)
#Definition of peaks
peaksUp=np.where(d1 > mean1+4*std1)
peaksDown=np.where(d1 < mean1-4*std1)
#Appending the peaks in a list
peaks=np.hstack((peaksUp,peaksDown))
#Sorting them
peaks = np.sort(peaks)[0]
#print((peaks))
#Clustering so the peaks only show up once per pertubation
peaksClustered=np.array([])
for j in range(len(peaks)-1):
    #print((peaks[j]))
    if abs(peaks[j]-peaks[j+1])>150:
        peaksClustered=np.append(peaksClustered,int(peaks[j]))
        #print(int(peaks[j]))

print(peaksClustered)
#Plotting each of the peaks' neighbourhood
for i in range(len(peaksClustered)):
    print(i)
    plt.figure()
    plt.plot(d1[int(peaksClustered[i]-120):int(peaksClustered[i]+550)])
    plt.xlabel(r'Time [$\mu$s]', fontdict=font)
    plt.ylabel(r'displacement [mm]', fontdict=font)
    plt.show()


# In[58]:


#Analysis of the individual peak
Peaknr = 6 #which of the peaks in the clustered peaks
left_edge = 120 #distance from peak to the left part of the image
Right_edge= 550#distance from peak to the right part of the image
#testx = d1[int(peaksClustered[Peaknr]-left_edge):int(peaksClustered[Peaknr]+Right_edge)]
testy = d1[int(peaksClustered[Peaknr]-left_edge):int(peaksClustered[Peaknr]+Right_edge)]

testx=np.arange(0,len(testy)/1000,1/1000
                )
#Zeroing time
t0 = testx[0]
testx -=t0
#Zeroing average
m = np.mean(testy)
testy -=m
#Function we want to fit
def func_power(x, a, b, c, d): #a, b, c, d, e, f
    #return(a+b*(x-t0))
    return(a*(x-b)**(c)*np.sin(189*x-1.8))
def func_exp(x, a, b, c, d): #a, b, c, d, e, f
    #return(a+b*(x-t0))
    return(a*np.exp(-b*x)*np.sin(c*x+d))    
#fitting 
params, _ = curve_fit(func_exp, testx, testy, maxfev=5000)
#params, _ = curve_fit(func_power, testx, testy, maxfev=5000)
#parameters we fit
a, b, c, d = params[0], params[1], params[2], params[3]#, params[4], params[5]
#yfit1 = a*np.exp(-b*testx)*np.sin(c*testx+d)
yfit1 = func_exp(testx, a, b, c, d)
#yfit1 = func_power(testx, a, b, c, d)

print("params: ", params) 
print('{0:.3g}*np.exp(-{1:.3g}*x)*np.sin({2:.3g}*x+{3:.3g})'.format(a,b,c,d))

plt.figure()
plt.title('Axial Damping', fontdict=font)
plt.plot(testx,testy,label='Experimental results')
plt.plot(testx,yfit1,label='Fit: {0:.3g}*np.exp(-{1:.3g}*x)*np.sin({2:.3g}*x+{3:.3g})'.format(a,b,c,d))
plt.xlabel(r'Time [$\mu$s]', fontdict=font)
plt.ylabel(r'displacement [mm]', fontdict=font)
plt.legend()
plt.grid()
#plt.plot(testx,yfit1)

plt.show()
#dataframe=pd.DataFrame(testx-testx[0], columns=['a'])#
#dataframe.to_csv(r'test.txt', header=None, index=None, sep=' ', mode='a')
#print(testy)
plt.plot(plt.plot(np.fft.fft(testy)))

# In[56]:


df = pd.read_csv ('R&D/Experiment_Damping/ExperimentResults_exp-1.csv')

#Making numpy array
q=df.to_numpy()

w=np.zeros(np.shape(q))
w[:,0] = q[:,-1]
for i in range(len(w[0,:])-1):
    w[:,i+1] = q[:,i]
print(w)
df2 =pd.DataFrame(w,index=None,columns=['Timesteps','exp1-1-X-10000SamplesPrSec_1.2mmFP','exp1-2-X-10000SamplesPrSec_1.2mmFP','exp1-3-X-10000SamplesPrSec_1.2mmFP','exp1-4-Y-10000SamplesPrSec_1.2mmFP','exp1-5-Z-10000SamplesPrSec_1.2mmFP'])
df2.to_csv("R&D/Experiment_Damping/Testcsv.csv",index=False)

