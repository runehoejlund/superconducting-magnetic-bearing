# -*- coding: utf-8 -*-
# %%
"""
Skylab Experiments:
Measurement of restoring force vs displacement

Previously this file was named: "skylab_restoring_force_experiment_5.py".

Experiment 5:

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from find_plateaus import find_plateaus
from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "font.size": 16})

axis = 'r'
m = 0.752
filename = 'force-journal-r'

origData = pd.read_csv("./data/journal-SMB/skylab_experiment_5_radial_pm_0,2mm.csv")
origTimestamps = pd.read_csv("./data/journal-SMB/skylab_experiment_5_radial_pm_0,2mm_timestamps.csv")

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
xForce = -np.array(origData["y-force"])

# %%
x = -np.array(origTimestamps["x"])
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

# plt.figure()
# plt.plot(time[start:end],d2F[start:end])
# plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
# plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
# plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')


# %%
# plt.figure()
# plt.plot(time[start:end], xForce[start:end])
# plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
# # plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
# # plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
# plt.xlabel('time [s]')
# plt.ylabel('Restoring force [N]')

# %%

# plt.figure()
# plt.plot(x,force,'-+')
# plt.xlabel('x [mm]')
# plt.ylabel('Restoring force [N]')
# plt.savefig('./plots/experiment-5-force.pdf')

# %%
def correct_for_N_vaporisation(t, xForce):
    # Method 1
    dF = np.gradient(xForce)
    mean_plateau_descent = np.array([np.mean(dF[int(np.floor(i+0.2*(j-i))):int(np.floor(i+0.8*(j-i)))]) for (i,j) in plateaus])
    N_loss_per_frame = np.mean(mean_plateau_descent)
    print("N loss per frame: " + str(N_loss_per_frame))
    
    a1 = 100*N_loss_per_frame # Nitrogen force loss per second
    F_corr = -a1*t
    # plt.figure()
    # plt.plot(t,F_corr)
    return xForce + F_corr

corrected_xForce = correct_for_N_vaporisation(time, xForce)
corrected_force = np.array([np.mean(corrected_xForce[int(np.floor(i+0.2*(j-i))):int(np.floor(i+0.8*(j-i)))]) for (i,j) in plateaus])

# plt.figure()
# plt.plot(time,corrected_xForce)
# plt.xlabel('time [s]')
# plt.ylabel('Corrected restoring force [N]')

f = plt.figure(figsize=(6,6))
plt.plot(x,corrected_force,'--x',linewidth=0.8, color = 'black', markersize=5, markerfacecolor='darkblue',markeredgecolor='k')
plt.grid(color='lightgrey', linestyle='-', linewidth=0.1)
f.get_axes()[0].annotate("", xy=(-0.02, 5.3), xytext=(0.03, 2), arrowprops=dict(arrowstyle="-|>"))
#f.get_axes()[0].annotate("", xy=(-0.05, 5.2), xytext=(-0.01, 0.5), arrowprops=dict(arrowstyle="-|>"))
f.get_axes()[0].annotate("", xy=(0.02, -3.3), xytext=(-0.03, 0), arrowprops=dict(arrowstyle="-|>"))
plt.xlabel('$r$ [mm]')
plt.ylabel('$F_r$ [N]')
plt.ylim([-22, 22])
plt.xlim([-0.25, 0.23])
plt.xticks(np.arange(-0.2, 0.21, 0.05), rotation=30, ha="right")
plt.title('Radial Force of Journal SMB Rotor')
plt.tight_layout()
plt.savefig('./plots/' + filename + '-corrected.pdf')
plt.savefig('./plots/' + filename + '-corrected.png')
plt.show()

# %% Calculate Spring constant
kx = - np.mean(np.gradient(corrected_force[0:7],x[0:7]))
print("stiffness kx = " + str(round(kx)) + " N/mm")
print("stiffness kx = " + str(round(kx * 10)) + " N/(10 mm)")

# Calcalate expected eigenfrequency
m = 0.6 # Mass of bearing in kg
omega_0 = np.sqrt(1000*kx/m)
f_0 = omega_0/(2*np.pi)
print("Expected eigenfrequency omega_0: " + str(omega_0) + " 1/s")
print("Expected eigenfrequency f_0: " + str(f_0) + " Hz")

# %% Calculate Energy dissipated
hysteresis_start = np.argmin(x)
delta_E = - np.trapz(corrected_force[hysteresis_start:],x[hysteresis_start:])
print("Energy Dissipated: " + str(round(delta_E,2)) + " N * mm")

# Calculate equivallent damping (Daniel J. Inmann, Engineering Vibrations Ch. 2.7)
# NOTE: I'm not sure, we should use omega_0 - maybe it should be omega
A = 0.2 # Amplitude = max(z)
c_eq = 1000* delta_E/(np.pi*omega_0*(A**2))
print("Equivallent damping. c_eq: " + str(round(c_eq,2)) + " N * s / m")

gamma_eq = c_eq/2*m
print("Equivallent damping. gamma_eq: " + str(round(gamma_eq,2)) + " 1/s")

zeta_eq = c_eq/(2*np.sqrt(kx*m))
print("Equivallent damping. zeta_eq: " + str(round(zeta_eq,2)))

# %%
modal_df = pd.DataFrame(
    [np.round([kx, f_0, omega_0, gamma_eq, zeta_eq, c_eq],3)],
    columns=['k_' + axis + ' [kN/m]', 'f_0 [Hz]', 'omega_0 [1/s]', 'gamma_eq [1/s]', 'zeta_eq', 'c_eq [kg/s]'])
modal_df.to_csv('./plots/' + filename + '.csv')
