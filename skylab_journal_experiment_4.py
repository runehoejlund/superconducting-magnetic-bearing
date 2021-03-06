# -*- coding: utf-8 -*-
"""
Skylab Experiments:
Measurement of restoring force vs displacement

Previously this file was named: "skylab_restoring_force_experiment_4.py".

Experiment 4:

"""
# %%
import numpy as np
import pandas as pd
from find_plateaus import find_plateaus
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "font.size": 16})

axis = 'z'
m = 0.752 # Mass of bearing in kg
filename = 'force-journal-z'

##
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

# plt.plot(time[start:end],d2F[start:end])
# plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
# plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
# plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
# plt.show()

# %%
# plt.plot(time[start:end], zForce[start:end])
# # plt.vlines(plateaus[(plateaus >= start) & (plateaus < end)]/100, ymin = ymin(), ymax = ymax(), colors='purple')
# # plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
# # plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
# plt.xlabel('time [s]')
# plt.ylabel('Restoring force [N]')
# plt.show()

# %%
# plt.plot(z,force,'-o',linewidth=0.8, markersize=4, markerfacecolor='red',markeredgecolor='black')
# plt.grid()
# #plt.arrow(0,5, -0.02,2,fc="k", ec="k",head_width=0.01, head_length=0.5)
# # f.get_axes()[0].annotate("", xy=(-0.02, 10.5), xytext=(0.03, 6), arrowprops=dict(arrowstyle="->"))
# # f.get_axes()[0].annotate("", xy=(0.02, -5), xytext=(-0.03, -1), arrowprops=dict(arrowstyle="->"))
# plt.xlabel('z [mm]', **hfont)
# plt.ylabel('Restoring force [N]')
# plt.savefig('./plots/experiment-4-force.pdf')
# plt.show()

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
    # plt.plot(t,F_corr)
    return zForce + F_corr

corrected_zForce = correct_for_N_vaporisation(time, zForce)
corrected_force = np.array([np.mean(corrected_zForce[int(i):int(j)]) for (i,j) in plateaus])
corrected_force = corrected_force - corrected_force[0]

# plt.plot(time,corrected_zForce)
# plt.xlabel('time [s]')
# plt.ylabel('Corrected restoring force [N]')
# plt.show()

# %%
# Load Simulated Data
df_sim = pd.read_csv("./data/journal-SMB-simulation/restoring_force.txt",header=7,sep="\s+")
z_sim = np.array(df_sim['r'])
force_sim = np.array(df_sim['Height'])

# %%
f = plt.figure(figsize=(6,6))
plt.plot(z,corrected_force,
    '--x', linewidth=0.8, color = 'black', markersize=5,markerfacecolor='darkblue',markeredgecolor='k',
    label='Experimental')
plt.plot(z_sim,force_sim,
    '-',linewidth=1.2, color = 'royalblue', markersize=5, markerfacecolor='darkblue',markeredgecolor='k',
    label='Simulated')
f.get_axes()[0].annotate("", xy=(-0.02, 8.5), xytext=(0.03, 4), arrowprops=dict(arrowstyle="-|>"))
#f.get_axes()[0].annotate("", xy=(-0.05, 5.2), xytext=(-0.01, 0.5), arrowprops=dict(arrowstyle="-|>"))
f.get_axes()[0].annotate("", xy=(0.02, -6), xytext=(-0.03, -2), arrowprops=dict(arrowstyle="-|>"))

plt.grid(color='lightgrey', linestyle='-', linewidth=0.1)
plt.legend()
plt.xlabel('$z$ [mm]')
plt.ylabel('$F_z$ [N]')
plt.ylim([-22, 22])
plt.xlim([-0.25, 0.23])
plt.xticks(np.arange(-0.2, 0.21, 0.05), rotation=30, ha="right")
plt.title('Lift Force of Journal SMB Rotor')
plt.tight_layout()
plt.savefig('./plots/' + filename + '-corrected.pdf')
plt.savefig('./plots/' + filename + '-corrected.png')
plt.show()

# %% Calculate Spring constant
kz = - np.mean(np.gradient(corrected_force[0:7],z[0:7]))
print("stiffness kz = " + str(round(kz)) + " N/mm")
print("stiffness kz = " + str(round(kz * 10)) + " N/(10 mm)")

# Calcalate expected eigenfrequency
omega_0 = np.sqrt(1000*kz/m)
f_0 = omega_0/(2*np.pi)
print("Expected eigenfrequency omega_0: " + str(omega_0) + " 1/s")
print("Expected eigenfrequency f_0: " + str(f_0) + " Hz")

# %% Calculate Energy dissipated
hysteresis_start = np.argmin(z)
delta_E = - np.trapz(corrected_force[hysteresis_start:],z[hysteresis_start:])
print("Energy Dissipated: " + str(round(delta_E,2)) + " N * mm")

# Calculate equivallent damping (Daniel J. Inmann, Engineering Vibrations Ch. 2.7)
# NOTE: I'm not sure, we should use omega_0 - maybe it should be omega
A = 0.2 # Amplitude = max(z)
c_eq = 1000*delta_E/(np.pi*omega_0*(A**2))
print("Equivallent damping. c_eq: " + str(round(c_eq,2)) + " N * s / m")

gamma_eq = c_eq/2*m
print("Equivallent damping. gamma_eq: " + str(round(gamma_eq,2)) + " 1/s")

zeta_eq = c_eq/(2*np.sqrt(kz*m))
print("Equivallent damping. zeta_eq: " + str(round(zeta_eq,2)))

# %%
modal_df = pd.DataFrame(
    [np.round([kz, f_0, omega_0, gamma_eq, zeta_eq, c_eq],3)],
    columns=['k_' + axis + ' [kN/m]', 'f_0 [Hz]', 'omega_0 [1/s]', 'gamma_eq [1/s]', 'zeta_eq', 'c_eq [kg/s]'])
modal_df.to_csv('./plots/' + filename + '.csv')