#%%
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

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

plt.figure()
plt.title('Radial Bearing oscillation - single oscillation')
plt.plot(t_one, y_one)
plt.xlabel('time [s]')
plt.ylabel('displacement [mm]')
plt.savefig('./single-oscillation.png')
plt.show()

plt.figure()
plt.title('FFT - single oscillation')
f = np.fft.fft(y_one)
freq = np.fft.fftfreq(t_one.shape[-1], d=1/fs)
plt.plot(freq[5:int(N_one/2)], np.abs(f[5:int(N_one/2)]))
plt.xlabel('frequency [Hz]')
plt.savefig('./fft.png')
plt.show()

# %%
# Find peaks
F = abs(f[5:int(N_one/2)])
Freq = freq[5:int(N_one/2)]
peaks, _ = find_peaks(F, prominence=1, width=1)
plt.plot(Freq, F)
plt.plot(Freq[peaks], F[peaks], "x")
plt.show()

f_eig = Freq[peaks][0]
print("eigenfrequency: " + str(f_eig) + " Hz")

# %%
# Fit curve

from scipy.optimize import curve_fit

def func(tt, f_0, g, phi, A):
    return A * np.exp(- g * tt) * np.cos(2*np.pi*f_0 * tt - phi)

p0 = (f_eig, 4, 0, np.max(y_one))
p, pcov = curve_fit(func, t_one[:int(N_one/4)], y_one[:int(N_one/4)])
print(p)
f_fitted, gamma_fitted, _, _ = p
print("f_fitted: " + str(f_fitted) + " Hz")
print("gamma_fitted: " + str(gamma_fitted) + " 1/s")

plt.plot(t_one, y_one)
plt.plot(t_one, func(t_one, *p), 'r-', label='fit: f_0=%5.3f, g=%5.3f, phi=%5.3f, A=%5.3f' % tuple(p))
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
