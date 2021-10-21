#%%
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# from scipy.fftpack import fft

fs = 1000 # sample frequency
df = pd.read_csv('Damping_axial_succes_slightly_off_center_1.csv',header=0)
Y = df["y"].to_numpy()
T = np.linspace(0, (len(Y)-1)/fs, len(Y))

N = 2**15
start = 0
end = start + N
t = T[start:end]
z = Y[start:end]

# standardise data (around equilibrium)
y = z - np.mean(z)
#%%
# Subtract linear bias

import numpy as np
from sklearn.linear_model import LinearRegression
model = LinearRegression()
t_in = t.reshape((-1, 1))
model.fit(t_in, z)

plt.plot(t, z)
plt.plot(t,model.predict(t_in),'r',label="Linear regression: " + str(round(model.coef_[0],3)) + " t + " + str(round(model.intercept_,3)))
plt.legend()
plt.xlabel('time [s]')
plt.ylabel('displacement [mm]')
plt.savefig('displacement.png')
plt.show()

# %%
# Plot the corrected data
y = z - model.predict(t_in)
plt.title('Radial Bearing oscillation')
plt.plot(t, y)
plt.xlabel('time [s]')
plt.ylabel('displacement [mm]')
plt.show()


#%%
N_one = 2**10
start_one = 8100
end_one = start_one + N_one
t_one = t[start_one:end_one]
y_one = y[start_one:end_one]

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
plt.plot(freq[1:int(N_one/2)], np.abs(f[1:int(N_one/2)]))
plt.xlabel('frequency [Hz]')
plt.savefig('./fft.png')
plt.show()

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
plt.show()
#%%
# Short Time fourier Transform
f, time, Zxx = signal.stft(y, fs=fs, nperseg=2**10)
STFT = np.abs(Zxx)
Fxx = np.sum(STFT,axis=1)
print(np.shape(Fxx))
print(np.shape(STFT))
print(np.max(STFT))

plt.plot(f[4:],Fxx[4:])
plt.show()

#%%
plt.pcolormesh(time, f[:int(round(len(f)/6))], np.abs(Zxx[:int(round(len(f)/6)),:]), vmin=np.quantile(STFT, 0.75), vmax=np.quantile(STFT, 0.95), shading='gouraud')
plt.title('SFTP Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar()
plt.show()

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
