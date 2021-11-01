# -*- coding: utf-8 -*-
#%%
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib import rcParams
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.sans-serif": ["Computer Modern Roman"],
    "font.size": 16})

# Import data
axis = 'r'
filename = 'experimental_modal_analysis_journal_r'
title = 'Radial oscillation of journal SMB Rotor'
m = 0.752 # Mass of rotor in kg

fs = 1000 # sample frequency
df = pd.read_csv('./data/' + filename + '.csv', header=0, names=["y"])

Y = df["y"].to_numpy()
T = np.linspace(0, (len(Y)-1)/fs, len(Y))

N = len(Y)
start = 0
end = start + N
t = T[start:end]
z = Y[start:end]

# standardise data (around equilibrium)
y = z - np.mean(z)

# %%
plt.plot(t, y)
plt.xlabel('time [s]')
plt.ylabel('$' + axis + '$ [mm]')
# plt.savefig('./plots/' + filename + '-displacement-all-impacts.png')
# plt.show()

#%% Pick out single oscillation
N_one = 2**12
start_one = 24350
end_one = start_one + N_one
t_one = t[start_one:end_one] - t[start_one]
y_one = y[start_one:end_one]
y_one = y_one - np.mean(y_one)

# Fourier transform (FFT on single oscillation)
f = np.fft.fft(y_one)
freq = np.fft.fftfreq(t_one.shape[-1], d=1/fs)  

# %%
# Find peaks on Fourier Spectrum
F = abs(f[5:int(N_one/2)])
Freq = freq[5:int(N_one/2)]
peaks, _ = find_peaks(F, prominence=0.8, width=1)
big_peaks = peaks[np.argsort(-F[peaks])[:2]]
f_fft = Freq[big_peaks[0]]
print("eigenfrequency: " + str(f_fft) + " Hz")

# Plot Fourier Spectrum
plt.figure(figsize=(6,4))
plt.plot(Freq, F, '-k', linewidth=1.0)
plt.grid(color='grey', linestyle='-', linewidth=0.2)
plt.xticks(np.hstack([np.arange(0, 501, 100), f_fft]))
plt.ylim(0,int(np.max(F)+2))
plt.xlabel('$f_n \,\, \mathrm{[Hz]}$')
for big_peak in big_peaks:
    p_x = Freq[big_peak]
    p_y = F[big_peak]
    plt.plot(p_x, p_y, "kx", markersize=8)
    plt.annotate(str(np.round(p_x,1)) + ' Hz', xy=(p_x + 10, p_y))
# plt.plot(Freq[peaks], F[peaks], "x", markerfacecolor='darkblue', markersize=10)
plt.title('Fourier Spectrum')
plt.tight_layout()
plt.savefig('./plots/' + filename + '-fft.pdf')
# plt.show()

# %%
# Fit curve

from scipy.optimize import curve_fit

def func(tt, g, phi, A):
    f1 = f_fft
    return A * np.exp(- g * tt) * np.cos(2 * np.pi * f1 * tt - phi)
f_fitted = f_fft

p0 = (4, 0, np.max(y_one))
p, pcov = curve_fit(func, t_one[:int(N_one/4)], y_one[:int(N_one/4)])
print(p)
gamma_fitted, _, A_fitted = p
omega_fitted = 2*np.pi*f_fitted
omega_0 = np.sqrt(omega_fitted**2 + gamma_fitted**2)
f_0 = omega_0/(2*np.pi)
zeta_fitted = gamma_fitted/omega_0
c_fitted = 2*m*gamma_fitted
k = m * omega_0**2

print("omega_fitted: " + str(omega_fitted) + " 1/s")
print("f_fitted: " + str(f_fitted) + " Hz")
print("omega_0: " + str(omega_0) + " 1/s")
print("f_0: " + str(f_0) + " Hz")
print("gamma_fitted: " + str(gamma_fitted) + " 1/s")
print("zeta_fitted: " + str(zeta_fitted))
print("c_fitted: " + str(c_fitted) + " kg/s")

plt.figure(figsize=(6,4))
plt.grid(color='grey', linestyle='-', linewidth=0.2)
plt.plot(t_one[:int(N_one/6)], y_one[:int(N_one/6)],'-.x',linewidth=0.8, color = 'lightgrey', markersize=1.2, markerfacecolor='darkblue',markeredgecolor='black')
plt.plot(t_one[:int(N_one/6)], func(t_one[:int(N_one/6)], *p), '-',linewidth=1.2, color = 'royalblue', markersize=1, markerfacecolor='darkblue', markeredgecolor='k')
plt.xlabel('t [s]')
plt.ylabel('$' + axis + '$ [mm]')
plt.title(title)
plt.legend(['Experimental data','$e^{-' + str(round(gamma_fitted,1)) + '\, t}\, ' + str(round(A_fitted,2)) +' \, \sin(2 \pi \cdot ' + str(round(f_fitted,1)) + '\, t - \phi)$'])
plt.tight_layout()
plt.savefig('./plots/' + filename + '-displacement.pdf')
plt.savefig('./plots/' + filename + '-displacement.pdf')
# plt.show()

# #%% Short Time fourier Transform for single impact
# f, time, Zxx = signal.stft(y_one, fs=fs, nperseg=2**10)
# STFT = np.abs(Zxx)

# plt.figure()
# plt.pcolormesh(time, f, np.abs(Zxx), vmin=np.quantile(STFT, 0.0), vmax=np.quantile(STFT, 0.99), shading='gouraud')
# plt.title('SFTP Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.colorbar()
# plt.savefig('./plots/' + filename + '-stfft.png')
# plt.show()

# #%% Short Time fourier Transform across all impacts
# # Short Time fourier Transform
# f, time, Zxx = signal.stft(y, fs=fs, nperseg=2**10)
# STFT = np.abs(Zxx)
# Fxx = np.sum(STFT,axis=1)
# print(np.shape(Fxx))
# print(np.shape(STFT))
# print(np.max(STFT))
# plt.plot(f[4:],Fxx[4:])
# plt.show()

# #%% Spectrogram plot
# plt.pcolormesh(time, f[:int(round(len(f)/6))], np.abs(Zxx[:int(round(len(f)/6)),:]), vmin=np.quantile(STFT, 0.75), vmax=np.quantile(STFT, 0.95), shading='gouraud')
# plt.title('SFTP Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.colorbar()
# plt.show()

# #%%
# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(t, y)

# plt.subplot(2,1,2)
# plt.pcolormesh(time, f, np.abs(Zxx), vmin=np.quantile(STFT, 0.92), vmax=np.quantile(STFT, 0.98), shading='gouraud')
# plt.title('SFTP Magnitude')
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [s]')
# plt.show()

# %% Export important data to csv
modal_df = pd.DataFrame(
    [np.round([k/1000, f_fft, f_fitted, omega_fitted, omega_0, f_0, gamma_fitted, zeta_fitted, c_fitted],3)],
    columns=['k [kN/m]', 'f_fft [Hz]', 'f_fitted [Hz]', 'omega_fitted [1/s]', 'omega_0 [1/s]', 'f_0 [Hz]', 'gamma_fitted [1/s]', 'zeta_fitted', 'c_fitted [kg/s]'])
modal_df.to_csv('./plots/' + filename + '.csv')

# %%
