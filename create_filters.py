import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
import os
from gabor_filter import gabor_filter

def hz2mel(f):
    """ Converts frequency from Hertz to Mel scale. """
    return 1127 * np.log(1 + f / 700)

def mel2hz(mel):
    """ Converts frequency from Mel scale to Hertz. """
    return 700 * (np.exp(mel / 1127) - 1)

def create_filter_bank(fs, M=12, size=1102):
    """ Creates a bank of Gabor filters based on the Mel scale. """
    mel_min = hz2mel(0)
    mel_max = hz2mel(fs / 2)
    mel_points = np.linspace(mel_min, mel_max, M + 1)
    hz_points = mel2hz(mel_points)
    
    filters_cos = []
    filters_sin = []
    params = []
    
    print(f"Generating {M} filters...")
    
    for i in range(M):
        start_hz = hz_points[i]
        end_hz = hz_points[i+1]
        l_i = end_hz - start_hz
        c_i = (start_hz + end_hz) / 2
        
        f_i = c_i / fs
        sigma_i = fs / l_i
        
        cos_h, sin_h = gabor_filter(size, sigma_i, f_i)
        
        filters_cos.append(cos_h)
        filters_sin.append(sin_h)
        params.append((f_i, sigma_i))

    return filters_cos, filters_sin, params

def plot_results(filters_cos, filters_sin, fs, student_id, save_dir="."):
    """
    Plots and saves figures using the strict naming convention:
    id_gabor_cos.png, id_gabor_sin.png, id_spectru_filtre.png
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    size = len(filters_cos[0])
    t = np.arange(size)
    
    # --- 1. Save Cos Filter Figure ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, filters_cos[0])
    plt.title(f"Gabor Filter (Cos) - First Segment\nfs={fs}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    filename_cos = f"{student_id}_gabor_cos.png"
    plt.savefig(os.path.join(save_dir, filename_cos))
    plt.close()
    
    # --- 2. Save Sin Filter Figure ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, filters_sin[0])
    plt.title(f"Gabor Filter (Sin) - First Segment\nfs={fs}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
    filename_sin = f"{student_id}_gabor_sin.png"
    plt.savefig(os.path.join(save_dir, filename_sin))
    plt.close()

    # --- 3. Save Spectrum Figure ---
    plt.figure(figsize=(12, 6))
    for h in filters_cos:
        fft_vals = scipy.fft.fft(h)
        half_size = size // 2
        magnitude = np.abs(fft_vals[:half_size])
        plt.plot(magnitude)
        
    plt.title("Gabor Filters Spectrum")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Magnitude")
    plt.grid(True, alpha=0.3)
    
    filename_spec = f"{student_id}_spectru_filtre.png"
    plt.savefig(os.path.join(save_dir, filename_spec))
    plt.close()
    
    print(f"Saved images to '{save_dir}':")
    print(f" - {filename_cos}")
    print(f" - {filename_sin}")
    print(f" - {filename_spec}")
