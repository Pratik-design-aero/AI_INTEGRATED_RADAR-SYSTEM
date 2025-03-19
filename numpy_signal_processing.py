import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print("Starting Script")

# Read the radar signal data
signal_data = pd.read_csv("Your File Name.csv")  
signal = signal_data.iloc[:, 11].values.astype(float)  # Convert to float

# Generate a time array based on the length of the signal
time = np.linspace(0, 1, len(signal))

# Generate Gaussian noise with the same length as the signal
noise = np.random.normal(0, 0.5, len(signal))

# Add noise to the signal
radar_signal = signal + noise

print("Signal Generated........")

mean_value = np.mean(radar_signal)
rms_value = np.sqrt(np.mean(radar_signal**2))
std_dev = np.std(radar_signal)

print(f"Mean value: {mean_value:.3f}, RMS : {rms_value:.3f}, Standard Deviation: {std_dev:.3f}")

# Smooth the noisy radar signal
window_size = 50
smoothed_signal = np.convolve(radar_signal, np.ones(window_size) / window_size, mode='valid')
smoothed_time = time[:len(smoothed_signal)]  # Adjust time array for smoothed signal

# Plot the original, noisy, and smoothed signals
plt.figure(figsize=(10, 6))
plt.plot(time, signal, label='Original Signal', linestyle='dashed')
plt.plot(time, radar_signal, label="Noisy Radar Signal", alpha=0.7)
plt.plot(smoothed_time, smoothed_signal, label="Smoothed Signal", alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Simulated Radar Signal with Noise and Smoothing")
plt.grid()
plt.show()