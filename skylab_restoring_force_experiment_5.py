# -*- coding: utf-8 -*-
"""
Skylab Experiments:
Measurement of restoring force vs displacement

Experiment 5:

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from find_plateaus import find_plateaus

origData = pd.read_csv("./data/skylab_experiment_5_radial_pm_0,2mm.csv")
origTimestamps = pd.read_csv("./data/skylab_experiment_5_radial_pm_0,2mm_timestamps.csv")

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
xForce = np.array(origData["y-force"])

# %%
x = np.array(origTimestamps["x"])
vp_start = 1.5 + np.array(origTimestamps["video_time_plateau_start"])
vp_end = 1.5 + np.array(origTimestamps["video_time_plateau_end"])

# %%
auto_plateaus, dF, d2F = find_plateaus(xForce,tolerance = 0.85, smoothing=30)
auto_plateaus = np.insert(auto_plateaus, 0, [vp_start[0]*100, vp_end[0]*100], axis=0)

plateaus = auto_plateaus

plateaus = np.empty((0,2))
temp = auto_plateaus
for video_plateau in vp_start:
    best = np.argmin(abs(video_plateau*100 - temp[:,0]))
    plateaus = np.append(plateaus,[temp[best]],axis=0)
    temp = np.delete(temp, best, axis=0)

force = np.array([np.mean(xForce[int(np.floor(i+0.2*(j-i))):int(np.floor(i+0.8*(j-i)))]) for (i,j) in plateaus])


# %% Plots
start = 0
end = 30000

plt.figure()
plt.plot(time[start:end],d2F[start:end])
plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')


# %%
plt.figure()
plt.plot(time[start:end], xForce[start:end])
plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
# plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
# plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
plt.xlabel('time [s]')
plt.ylabel('Restoring force [N]')

# %%

plt.figure()
plt.plot(x,force,'-+')
plt.xlabel('x [mm]')
plt.ylabel('Restoring force [N]')
plt.savefig('./plots/experiment-5-force.pdf')

# %%
def correct_for_N_vaporisation(t, xForce):
    # Method 1
    dF = np.gradient(xForce)
    mean_plateau_descent = np.array([np.mean(dF[int(np.floor(i+0.2*(j-i))):int(np.floor(i+0.8*(j-i)))]) for (i,j) in plateaus])
    N_loss_per_frame = np.mean(mean_plateau_descent)
    print("N loss per frame: " + str(N_loss_per_frame))
    
    a1 = 100*N_loss_per_frame # Nitrogen force loss per second
    F_corr = -a1*t
    plt.figure()
    plt.plot(t,F_corr)
    return xForce + F_corr

corrected_xForce = correct_for_N_vaporisation(time, xForce)
corrected_force = np.array([np.mean(corrected_xForce[int(np.floor(i+0.2*(j-i))):int(np.floor(i+0.8*(j-i)))]) for (i,j) in plateaus])

plt.figure()
plt.plot(time,corrected_xForce)
plt.xlabel('time [s]')
plt.ylabel('Corrected restoring force [N]')

plt.figure()
plt.plot(x,corrected_force,'-+')
plt.xlabel('x [mm]')
plt.ylabel('Radial force [N]')
plt.title('Radial Force vs. Radial Displacement')
plt.savefig('./plots/experiment-5-corrected-force.pdf')
plt.savefig('./plots/experiment-5-corrected-force.png')

# %% Calculate Spring constant
kx = np.mean(np.gradient(corrected_force[0:5],x[0:5]))
print("stiffness kz = " + str(round(kx)) + " N/mm")
