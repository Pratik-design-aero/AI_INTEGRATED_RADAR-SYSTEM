import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

fs =5000 #sampling frequency
T =0.05 #signal duration
t=np.linspace(0,T,int(T*fs),endpoint=False) #time vector
f0=1e9 #1 GHz
c=3e8 #speed of light

#Simulate a moving target

v_target= 50 #m/s
fd=2*v_target*f0/c #Doppler shift
received_signal=np.sin(2*np.pi*(f0+fd)*t) #Signal towards RADAR

#transmitted radar signal

transmitted_signal=np.sin(2*np.pi*f0*t)

#plot signals
plt.figure(figsize=(12,5))
plt.plot(t[:500],transmitted_signal[:500],label="Transmitted Signal",linestyle="--")
plt.plot(t[:500],received_signal[:500],label="Received Signal")
plt.title("Radar Signal with Doppler Shift")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


# Compute FFT of received signal
fft_received = np.fft.fft(received_signal)
frequencies = np.fft.fftfreq(len(t), 1/fs)

# Extract Doppler Frequency
doppler_peak = frequencies[np.argmax(np.abs(fft_received))]  # Find peak frequency
velocity_estimate = (doppler_peak * c) / (2 * f0)  # Calculate velocity

# Plot FFT Spectrum
plt.figure(figsize=(12, 5))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_received[:len(frequencies)//2]), label="FFT of Received Signal")
plt.title("FFT - Extracting Doppler Shift")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

# Print Estimated Velocity
print(f"Estimated Velocity: {velocity_estimate:.2f} m/s")
