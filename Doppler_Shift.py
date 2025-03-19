import numpy as np
import matplotlib.pyplot as plt

#Radar Parameters
fs =5000 # Sampling frequency
T =0.01 #Signal duration
t=np.linspace(0,T,int(T*fs),endpoint=False) # Time vector

#Original Radar Frequency

f0=1e9 # 1 GHz
c=3e8 # Speed of light

#Target moving towards RADAR
v_target= 50 # m/s
fd=2*v_target*f0/c # Doppler shift
signal_towards=np.sin(2*np.pi*(f0+fd)*t) # Signal towards RADAR

#Target moving away from RADAR
v_target_awar= -50 # m/s
fd_away=(2*v_target_awar*f0)/c # Doppler shift
signal_away=np.sin(2*np.pi*(f0+fd_away)*t) # Signal away from RADAR

#Plot the signals

plt.figure(figsize=(12,5))
plt.plot(t[:500],signal_towards[:500],label="Target moving towards RADAR")
plt.plot(t[:500],signal_away[:500],label="Target moving away from RADAR")
plt.title("Doppler Shift in Radar Signal")   
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()


# Compute FFT of Doppler Shifted Signals
fft_towards = np.fft.fft(signal_towards)
fft_away = np.fft.fft(signal_away)
frequencies = np.fft.fftfreq(len(t), 1/fs)

# Plot FFT Magnitude Spectrum
plt.figure(figsize=(12, 5))
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_towards[:len(frequencies)//2]), label="Towards (+50 m/s)")
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_away[:len(frequencies)//2]), label="Away (-50 m/s)")
plt.title("FFT - Doppler Frequency Shift")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.show()
