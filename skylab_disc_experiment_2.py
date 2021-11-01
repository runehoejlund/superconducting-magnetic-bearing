# -*- coding: utf-8 -*-
"""
Skylab Experiments on June 30th 2021:
Measurement of restoring force vs displacement for disc bearing.

Experiment 2:

"""
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from find_plateaus import find_plateaus
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "font.size": 16})

axis = 'z'
m = 1.526 # Mass of bearing in kg
filename = 'force-disc-z'

origData = pd.read_csv("./data/disc-SMB/skylab_disc-experiment_2_axial_pm_0,2mm.csv")
origTimestamps = pd.read_csv("./data/disc-SMB/skylab_disc-experiment_2_axial_pm_0,2mm_timestamps.csv")

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
zForce = - np.array(origData["z-force"]) + np.array(origData["z-force"])[0]

z = np.array(origTimestamps["z"])
vp_start = np.array(origTimestamps["video_time_plateau_start"])
vp_end = np.array(origTimestamps["video_time_plateau_end"])

auto_plateaus, dF, d2F = find_plateaus(zForce, min_length=100, tolerance=0.75, smoothing=25)
auto_plateaus = np.insert(auto_plateaus, 0, [vp_start[0]*100, vp_end[0]*100], axis=0)

# plateaus = auto_plateaus  
plateaus = np.empty((0,2))
temp = auto_plateaus
for video_plateau in vp_start:
    best = np.argmin(abs(video_plateau*100 - temp[:,0]))
    plateaus = np.append(plateaus,[temp[best]],axis=0)
    temp = np.delete(temp, best, axis=0)

force = np.array([np.mean(zForce[int(i):int(j)]) for (i,j) in plateaus])

# %% Plots
start = 0
end = 6000

p_start = plateaus[:,0]
p_end = plateaus[:,1]

plt.figure()
plt.plot(time[start:end],d2F[start:end])
plt.vlines(p_start[(p_start >= start) & (p_start < end)]/100, ymin = ymin(), ymax = ymax(), colors='lime')
plt.vlines(p_end[(p_end >= p_end) & (p_end < end)]/100, ymin = ymin(), ymax = ymax(), colors='orangered')
plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
# plt.show()

# %%
start = 0
end = 56400
plt.figure()
plt.plot(time[start:end], zForce[start:end])
plt.vlines(p_start[(p_start >= start) & (p_start < end)]/100, ymin = ymin(), ymax = ymax(), colors='lime')
plt.vlines(p_end[(p_end >= p_end) & (p_end < end)]/100, ymin = ymin(), ymax = ymax(), colors='orangered')
# plt.vlines(vp_start[(vp_start >= start/100) & (vp_start < end/100)], ymin = ymin(), ymax = ymax(), colors='g')
# plt.vlines(vp_end[(vp_end >= start/100) & (vp_end < end/100)], ymin = ymin(), ymax = ymax(), colors='r')
# plt.show()

# %%

plt.figure()
plt.plot(z[abs(z) < 0.220],force[abs(z) < 0.220],'-+')
plt.xlabel('z [mm]')
plt.ylabel('Restoring force [N]')
plt.savefig('./plots/disc-experiment-2-force.pdf')
# plt.show()

# %%

f = plt.figure(figsize=(6,6))
plt.plot(z[abs(z) < 0.2],force[abs(z) < 0.2],'--x',linewidth=0.8, color = 'black', markersize=5, markerfacecolor='darkblue',markeredgecolor='k')
plt.grid(color='lightgrey', linestyle='-', linewidth=0.1)
f.get_axes()[0].annotate("", xy=(0.08, 10), xytext=(0.12, 4.5), arrowprops=dict(arrowstyle="-|>"))
#f.get_axes()[0].annotate("", xy=(-0.05, 5.2), xytext=(-0.01, 0.5), arrowprops=dict(arrowstyle="-|>"))
f.get_axes()[0].annotate("", xy=(0.02, -7.3), xytext=(-0.01, -2.3), arrowprops=dict(arrowstyle="-|>"))
plt.xlabel('$z$ [mm]')
plt.ylabel('$F_z$ [N]')
plt.ylim([-25, 50])
plt.xlim([-0.22, 0.22])
plt.xticks(np.arange(-0.2, 0.21, 0.05), rotation=30, ha="right")
plt.title('Axial Force of Disc SMB Rotor')
plt.tight_layout()
plt.savefig('./plots/' + filename + '.pdf')
plt.savefig('./plots/' + filename + '.png')
# plt.show()


# # %%
# def correct_for_N_vaporisation(t, zForce):   
#     ## Method 1: Find mean descent on plateaus
#     dF = np.gradient(zForce)
#     mean_plateau_descent = np.array([np.mean(dF[int(i):int(j)]) for (i,j) in plateaus])
#     N_loss_per_frame = np.mean(mean_plateau_descent)
#     a1 = 100*N_loss_per_frame # Nitrogen force loss per second
#     F_corr = -a1*t

#     ## Method 2: Find mean descent on first 50 seconds
#     # (where we are figuring out how to adjust the z-position)
#     # N_loss_per_frame = np.mean(np.gradient(zForce[1000:6000]))
#     # plt.figure()
#     # plt.plot(time[1000:6000],zForce[1000:6000],'-+')
#     # plt.xlabel('time [s]')
#     # plt.ylabel('Force [N]')
    
#     # Method 3: Find mean descent on 10 seconds right after refilling
#     N_loss_per_frame = np.mean(np.gradient(zForce[39500:40800]))
#     t1 = 386 # Refill start
#     t2 = 394 # Refill end
#     a1 = 100*N_loss_per_frame # Nitrogen force loss per second
#     a2 = 100*np.mean(np.gradient(zForce[int(t1*100):int(t2*100)])) # Nitrogen filling per second
#     F_corr = -np.piecewise(t, [t < t1, (t >= t1) & (t < t2), t >= t2],
#                      [lambda t: a1*t,
#                       lambda t: a1*t1 + a2*(t - t1),
#                       lambda t: a1*t1 + a2*(t2 - t1) + a1*(t - t2)])
#     plt.figure()
#     plt.plot(t,F_corr)
#     return zForce + F_corr

# corrected_zForce = correct_for_N_vaporisation(time, zForce)
# corrected_force = np.array([np.mean(corrected_zForce[int(i):int(j)]) for (i,j) in plateaus])

# plt.figure()
# plt.plot(z[abs(z) < 0.220],corrected_force[abs(z) < 0.220],'-+')
# plt.xlabel('z [mm]')
# plt.ylabel('Restoring force [N]')
# plt.savefig('./plots/disc-experiment-2-corrected-force.pdf')
# plt.show()
# # %% Calculate Spring constant
kz = np.abs(np.mean(np.gradient(force[0:5],z[0:5])))
print("stiffness kz = " + str(round(kz)) + " kN/m")

# %%
modal_df = pd.DataFrame(
    [np.round([kz],3)],
    columns=['k_' + axis + ' [kN/m]'])
modal_df.to_csv('./plots/' + filename + '.csv')
