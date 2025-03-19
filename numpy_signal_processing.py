import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sampling frequency (Hz)
T = 1  # Duration (sec)
t = np.linspace(0, T, fs, endpoint=False)  # Time vector

# Create a sine wave signal with Gaussian noise
f_signal = 50  # Frequency of the signal (Hz)
signal = np.sin(2 * np.pi * f_signal * t)  # Pure sine wave
noise = np.random.normal(0, 0.5, signal.shape)  # Gaussian noise
noisy_signal = signal + noise  # Noisy signal

# Compute FFT
fft_values = np.fft.fft(noisy_signal)
frequencies = np.fft.fftfreq(len(fft_values), 1/fs)

# Plot the time-domain signal
plt.figure(figsize=(12, 5))
plt.subplot(2,1,1)
plt.plot(t, noisy_signal)
plt.title("Noisy Signal in Time Domain")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

# Plot the frequency-domain signal (Magnitude Spectrum)
plt.subplot(2,1,2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_values[:len(fft_values)//2]))
plt.title("FFT - Frequency Domain Representation")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

from scipy.signal import butter, filtfilt

# Function to create a Butterworth Low-Pass Filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Apply filter
cutoff_freq = 60  # Cutoff frequency (Hz)
filtered_signal = butter_lowpass_filter(noisy_signal, cutoff_freq, fs)

# Plot the results
plt.figure(figsize=(12, 5))
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
plt.title("Low-Pass Filtering Effect")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Function to create a Butterworth High-Pass Filter
def butter_highpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

# Generate a noisy signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)  # Mixed frequencies
noise = np.random.normal(0, 0.2, signal.shape)  # Gaussian noise
noisy_signal = signal + noise

# Apply High-Pass Filter (removes low-frequency noise < 20Hz)
cutoff_freq = 20  # Cutoff frequency (Hz)
filtered_signal = butter_highpass_filter(noisy_signal, cutoff_freq, fs)

# Plot Results
plt.figure(figsize=(12, 5))
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(t, filtered_signal, label="High-Pass Filtered Signal", linewidth=2)
plt.title("High-Pass Filtering Effect")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

# Function to create a Butterworth Band-Pass Filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist Frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)

# Apply Band-Pass Filter (keeps 40Hz - 60Hz)
lowcut = 40  # Lower cutoff frequency (Hz)
highcut = 60  # Upper cutoff frequency (Hz)
filtered_signal = butter_bandpass_filter(noisy_signal, lowcut, highcut, fs)

# Plot Results
plt.figure(figsize=(12, 5))
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.5)
plt.plot(t, filtered_signal, label="Band-Pass Filtered Signal", linewidth=2)
plt.title("Band-Pass Filtering Effect (40Hz - 60Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.show()

