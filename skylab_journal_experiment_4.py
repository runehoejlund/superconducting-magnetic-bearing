# -*- coding: utf-8 -*-
"""
Skylab Experiments:
Measurement of restoring force vs displacement

Previously this file was named: "skylab_restoring_force_experiment_4.py".

Experiment 4:

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from find_plateaus import find_plateaus

origData = pd.read_csv("./data/journal-SMB/skylab_experiment_4_axial_pm_0,2mm.csv")
origTimestamps = pd.read_csv("./data/journal-SMB/skylab_experiment_4_axial_pm_0,2mm_timestamps.csv")

def getSeconds(timestring):
    tod = timestring.split(' ')[1].split(':')
    h = float(tod[0])
    m = float(tod[1])
    s = float(tod[2])
    return h*3600 + m*60 +s

def ymin():
    return plt.gca().get_ylim()[0]

def ymax():
    return plt.gca().get_ylim()[1]

seconds = np.array([getSeconds(t) for t in origData.Time])
t = seconds - seconds[0]
time = t
xForce = np.array(origData["x-force"])
zForce = -np.array(origData["z-force"])


z = np.array(origTimestamps["z"])
vp_start = np.array(origTimestamps["video_time_plateau_start"])
vp_end = np.array(origTimestamps["video_time_plateau_end"])

auto_plateaus, dF, d2F = find_plateaus(zForce)
auto_plateaus = np.insert(auto_plateaus, 0, [vp_start[0]*100, vp_end[0]*100], axis=0)

plateaus = np.empty((0,2))
temp = auto_plateaus
for video_plateau in vp_start:
    best = np.argmin(abs(video_plateau*100 - temp[:,0]))
    plateaus = np.append(plateaus,[temp[best]],axis=0)
    temp = np.delete(temp, best, axis=0)

force = np.array([np.mean(zForce[int(i):int(j)]) for (i,j) in plateaus])


# %% Plots
start = 0
end = 5000

plt.figure()
plt.plot(time[start:end],d2F[start:end])
plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')


# %%
plt.figure()
plt.plot(time[start:end], zForce[start:end])
# plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
# plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
# plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
plt.xlabel('time [s]')
plt.ylabel('Restoring force [N]')

# %%

plt.figure()
plt.plot(z,force,'-+')
plt.xlabel('z [mm]')
plt.ylabel('Restoring force [N]')
plt.savefig('./plots/experiment-4-force.pdf')

# %%
def correct_for_N_vaporisation(t, zForce):
    # Method 1
    dF = np.gradient(zForce)
    mean_plateau_descent = np.array([np.mean(dF[int(i):int(j)]) for (i,j) in plateaus])
    N_loss_per_frame = np.mean(mean_plateau_descent)
    print("N loss per frame: " + str(N_loss_per_frame))
    
    framerate = 100 # frames per second
    a1 = framerate*N_loss_per_frame # Nitrogen force loss per second
    F_corr = -a1*t
    plt.figure()
    plt.plot(t,F_corr)
    return zForce + F_corr

corrected_zForce = correct_for_N_vaporisation(time, zForce)
corrected_force = np.array([np.mean(corrected_zForce[int(i):int(j)]) for (i,j) in plateaus])

plt.figure()
plt.plot(time,corrected_zForce)
plt.xlabel('time [s]')
plt.ylabel('Corrected restoring force [N]')

plt.figure()
plt.plot(z,corrected_force,'-+')
plt.xlabel('z [mm]')
plt.ylabel('Lift force [N]')
plt.title('Lift Force vs. Axial Displacement')
plt.savefig('./plots/experiment-4-corrected-force.pdf')
plt.savefig('./plots/experiment-4-corrected-force.png')

# %% Calculate Spring constant
kz = np.mean(np.gradient(corrected_force[0:5],z[0:5]))
print("stiffness kz = " + str(round(kz)) + " N/mm")
