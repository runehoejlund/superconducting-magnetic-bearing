#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 15:34:53 2021

@author: runehoejlund
"""

# %%
dFdz = np.gradient(smoothedZForce)
plt.vlines(slopeStart[(slopeStart >= start/100) & (slopeStart < end/100)], ymin = -0.05, ymax = 0.05, colors='r')
peaks, peak_props = find_peaks(np.round(abs(dFdz),2), height=np.quantile(abs(dFdz),0.95),distance=400, width=20)

peaks_width = peak_props["widths"]
peaks_start = np.intc(np.round(peaks - peaks_width/2))

plt.figure(3)
plt.plot(time,dFdz)
plt.plot(peaks_start/100, dFdz[peaks_start], "x")
plt.plot(time[abs(dFdz) <= 0.005],dFdz[abs(dFdz) <= 0.005])



# %%
plt.figure(4)
plt.plot(time[start:end], zForce[start:end])
# plt.plot(time[(np.floor(abs(dFdz*100)) == 0)], zForce[(np.floor(abs(dFdz*100)) == 0)],color='purple')
plt.vlines(slopeStart[(slopeStart >= start/100) & (slopeStart < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
plt.vlines(peaks_start[(peaks_start >= start) & (peaks_start < end)]/100, ymin = ymin(), ymax = ymax(), colors='y')

# %%
plt.figure(7)
dF2dz2 = uniform_filter1d(np.gradient(dFdz),25)
plt.plot(time[start:end],dF2dz2[start:end])
plt.vlines(slopeEnd[(slopeEnd >= start/100) & (slopeEnd < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
plt.vlines(slopeStart[(slopeStart >= start/100) & (slopeStart < end/100)], ymin = ymin(), ymax = ymax(), colors='r')

# # %%
# peaks, peak_props = find_peaks(- smoothedZForce, plateau_size = 2)
# plateau_sizes = peak_props['plateau_sizes']
# for i in range(len(plateau_sizes)):
#     if plateau_sizes[i] > 1:
#         print('a plateau of size %d is found' % plateau_sizes[i])
#         print('its left index is %d and right index is %d' % (peak_props['left_edges'][i], peak_props['right_edges'][i]))
# plateauStart = peak_props['left_edges']
# plateauEnd = peak_props['right_edges']
# plt.figure(6)
# plt.plot(time[start:end], zForce[start:end])
# plt.vlines(plateauStart[(plateauStart >= start) & (plateauStart < end)]/100,ymin = ymin(), ymax = ymax(),color='k')
# plt.vlines(plateauEnd[(plateauEnd >= start) & (plateauEnd < end)]/100,ymin = ymin(), ymax = ymax(),color='b')
# plt.vlines(slopeEnd[(slopeEnd >= start/100) & (slopeEnd < end/100)], ymin = ymin(), ymax = ymax(), colors='r')

# %%

# def sectioning(force,time):
    
#     #smoothing of data
#     smoothedForce = uniform_filter1d(force, size=15)
    
#     #finding each of the sharp slopes
#     m = 50 # amount of points the slope is found over
#     step_limit = 0.05
    
#     index_list_s = [] #start points for each sudden peak
#     for i in range(len(smoothedForce)-m):
#         if smoothedForce[i+m]-smoothedForce[i]>step_limit:
#             index_list_s.append(i)
#     slopeStart=[]
#     for i in range(len(index_list_s)):
#         if index_list_s[i]-index_list_s[i-1]!=1:
#             slopeStart.append(int(index_list_s[i]))
            
#     index_list_e = [] #end points for each sudden peak
#     for i in range(len(smoothedForce)-m):
#         if smoothedForce[i]-smoothedForce[i-m]>step_limit:
#             index_list_e.append(i)
#     slopeEnd=[]
#     for i in range(len(index_list_e)-1):
#         if index_list_e[i+1]-index_list_e[i]!=1:
#             slopeEnd.append(int(index_list_e[i]))
#     slopeEnd.append(index_list_e[-1])
    
    
#     print(len(slopeEnd))
#     print(len(slopeStart))
#     if len(slopeEnd)!=len(slopeStart):
#         print('ERROR: not equal amount of starts and ends when indexing')
#         print(len(slopeEnd))
#         print(len(slopeStart))
#         return
    
#     #Now  we will sort each of the peaks so they come in pairs
#     L = 225 # minimum flat region between peaks
#     p = 0.5 # percentage of the data (where force is almost constant) it is averaged over
#     Index_list_flat_s_c2=[]
#     Index_list_flat_e_c2=[]
#     for i in range(len(slopeEnd)-1):
#         if slopeStart[i+1]-slopeEnd[i]>L:
#             Index_list_flat_s_c2.append(slopeEnd[i])
#             Index_list_flat_e_c2.append(slopeStart[i+1])
    
#     forceAvg=np.zeros(len(Index_list_flat_s_c2))
#     for i in range(len(Index_list_flat_s_c2)):
#             Index_End=math.floor((Index_list_flat_e_c2[i]-Index_list_flat_s_c2[i])*p+Index_list_flat_s_c2[i])
#             forceAvg[i] = np.average(force[Index_list_flat_s_c2[i]:Index_End])
#             if ((Index_list_flat_s_c2[i] >= start) and (Index_End < end)):
#                 plt.plot(time[Index_list_flat_s_c2[i]:Index_End],force[Index_list_flat_s_c2[i]:Index_End],'m')
#                 plt.vlines(Index_list_flat_s_c2[i]/100, ymin = ymin(), ymax = ymax(),color='k')

#     return forceAvg

# plt.figure(5)
# plt.plot(time[start:end],smoothedZForce[start:end])
# plt.vlines(slopeEnd[(slopeEnd >= start/100) & (slopeEnd < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
# sectioning(zForce,time)