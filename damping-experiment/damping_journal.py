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

fs = 1000 # sample frequency
df = pd.read_csv('damping_journal.csv', header=0, names=["y"])

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
plt.ylabel('displacement [mm]')
plt.savefig('journal-displacement.png')
plt.show()

#%%
N_one = 2**12
start_one = 24350
end_one = start_one + N_one
t_one = t[start_one:end_one] - t[start_one]
y_one = y[start_one:end_one]
y_one = y_one - np.mean(y_one)

f = np.fft.fft(y_one)
freq = np.fft.fftfreq(t_one.shape[-1], d=1/fs)

# %%
# Find peaks
F = abs(f[5:int(N_one/2)])
Freq = freq[5:int(N_one/2)]
peaks, _ = find_peaks(F, prominence=0.8, width=1)
f_eig = Freq[peaks][0]
print("eigenfrequency: " + str(f_eig) + " Hz")

plt.figure(figsize=(6,4))
plt.plot(Freq, F, '-k', linewidth=1.0)
plt.grid(color='grey', linestyle='-', linewidth=0.2)
plt.xticks(np.hstack([np.arange(0, 501, 100),f_eig]))
plt.xlabel('$f_n \,\, \mathrm{[Hz]}$')
# plt.plot(Freq[peaks], F[peaks], "x", markerfacecolor='darkblue', markersize=10)
plt.title('Fourier Spectrum')
plt.tight_layout()
plt.savefig('./plots/Journal_Damping_FFT.pdf')
plt.show()

# %%
# Fit curve

from scipy.optimize import curve_fit

def func(tt, omega_0, g, phi, A):
    return A * np.exp(- g * tt) * np.cos(omega_0 * tt - phi)

p0 = (f_eig/(2*np.pi), 4, 0, np.max(y_one))
p, pcov = curve_fit(func, t_one[:int(N_one/4)], y_one[:int(N_one/4)])
print(p)
omega_fitted, gamma_fitted, _, _ = p
f_fitted = 2*np.pi*omega_fitted
print("omega_fitted: " + str(omega_fitted) + " Hz")
print("f_fitted: " + str(f_fitted) + " Hz")
print("gamma_fitted: " + str(gamma_fitted) + " 1/s")

plt.figure(figsize=(6,4))
plt.grid(color='grey', linestyle='-', linewidth=0.2)
plt.plot(t_one[:int(N_one/6)], y_one[:int(N_one/6)],'-.x',linewidth=0.8, color = 'lightgrey', markersize=1.2, markerfacecolor='darkblue',markeredgecolor='black')
plt.plot(t_one[:int(N_one/6)], func(t_one[:int(N_one/6)], *p), '-',linewidth=1.2, color = 'royalblue', markersize=1, markerfacecolor='darkblue', markeredgecolor='k', label='fit: f_0=%5.3f, g=%5.3f, phi=%5.3f, A=%5.3f' % tuple(p))
plt.xlabel('t [s]')
plt.ylabel('$z$ [mm]')
plt.title('Oscillation of Journal SMB Rotor')
plt.legend(['Data','$A \, e^{-' + str(round(gamma_fitted,1)) + '\, t}\,\sin(' + str(round(omega_fitted,1)) + '\, t + \phi)$'])
plt.tight_layout()
plt.savefig('./plots/Journal_Damping.pdf')
plt.savefig('./plots/Journal_Damping.png')
plt.show()
# %%

#%%
# Short Time fourier Transform
f, time, Zxx = signal.stft(y_one, fs=fs, nperseg=2**10)
STFT = np.abs(Zxx)

plt.figure()
plt.pcolormesh(time, f, np.abs(Zxx), vmin=np.quantile(STFT, 0.0), vmax=np.quantile(STFT, 0.99), shading='gouraud')
plt.title('SFTP Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar()
plt.savefig('./stfft.png')
# plt.show()
#%%
# Short Time fourier Transform
f, time, Zxx = signal.stft(y, fs=fs, nperseg=2**10)
STFT = np.abs(Zxx)
Fxx = np.sum(STFT,axis=1)
print(np.shape(Fxx))
print(np.shape(STFT))
print(np.max(STFT))

plt.plot(f[4:],Fxx[4:])
# plt.show()

#%%
plt.pcolormesh(time, f[:int(round(len(f)/6))], np.abs(Zxx[:int(round(len(f)/6)),:]), vmin=np.quantile(STFT, 0.75), vmax=np.quantile(STFT, 0.95), shading='gouraud')
plt.title('SFTP Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar()
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

# %%
