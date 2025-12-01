import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import os
from mexican_hat_filter import mexican_hat_filter

def hz2mel(f):
    return 1127 * np.log(1 + f / 700)

def mel2hz(mel):
    return 700 * (np.exp(mel / 1127) - 1)

def create_custom_bank(fs, M=12, size=1102):
    """
    Creates a bank of Mexican Hat filters based on Mel scale center frequencies.
    """
    # 1. Define range in Mel scale
    mel_min = hz2mel(0)
    mel_max = hz2mel(fs / 2)
    mel_points = np.linspace(mel_min, mel_max, M + 1)
    hz_points = mel2hz(mel_points)
    
    filters = []
    
    print(f"Generating {M} custom filters (Mexican Hat)...")
    
    for i in range(M):
        # We use the center of the Mel segment as the target frequency
        start_hz = hz_points[i]
        end_hz = hz_points[i+1]
        center_hz = (start_hz + end_hz) / 2
        
        # Normalized frequency
        f_norm = center_hz / fs
        
        # For a Ricker wavelet, the peak frequency is approx f_peak = sqrt(2) / (pi * sigma)
        # Therefore, required sigma = sqrt(2) / (pi * f_norm)
        # We create a sigma that positions the filter at the correct frequency
        if f_norm > 0:
            sigma_custom = np.sqrt(2) / (np.pi * f_norm)
        else:
            # Fallback for 0Hz (though Mel start usually > 0 or handle gracefully)
            sigma_custom = size / 2 
            
        h = mexican_hat_filter(size, sigma_custom)
        filters.append(h)

    return filters

def plot_custom_results(filters, fs, student_id, save_dir="."):
    """
    Plots and saves the custom filter figures.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    size = len(filters[0])
    t = np.arange(size)
    
    # --- Plot 1: First Filter Time Domain ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, filters[0])
    plt.title(f"Mexican Hat Filter - First Segment\nfs={fs}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    filename_time = f"{student_id}_mexican_hat.png"
    plt.savefig(os.path.join(save_dir, filename_time))
    plt.close()

    # --- Plot 2: Spectrum of Filter Bank ---
    plt.figure(figsize=(12, 6))
    for h in filters:
        fft_vals = scipy.fft.fft(h)
        half_size = size // 2
        magnitude = np.abs(fft_vals[:half_size])
        plt.plot(magnitude)
        
    plt.title("Mexican Hat Filters Spectrum")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    
    filename_spec = f"{student_id}_spectru_mexican.png"
    plt.savefig(os.path.join(save_dir, filename_spec))
    plt.close()
    
    print(f"Saved custom images to '{save_dir}':")
    print(f" - {filename_time}")
    print(f" - {filename_spec}")
