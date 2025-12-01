import numpy as np

def gabor_filter(size, sigma, freq):
    """
    Creates a Gabor filter (cosine and sine variants).
    
    Parameters:
    size (int): The size of the filter (S).
    sigma (float): Standard deviation of the Gaussian envelope.
    freq (float): Frequency of the sinusoidal modulation.
    
    Returns:
    cos_h (numpy.ndarray): The Gabor filter modulated with cosine.
    sin_h (numpy.ndarray): The Gabor filter modulated with sine.
    """
    
    # 1. Define the time vector n from 0 to size-1 
    n = np.arange(0, size)
    
    # 2. Define the mean (mu) as center of the window (S/2)
    mu = size / 2
    
    # 3. Create the Gaussian envelope using Equation (2)
    # g(n) = (1 / (sigma * sqrt(2*pi))) * exp(-(n - mu)^2 / (2 * sigma^2))
    gaussian_env = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((n - mu)**2) / (2 * sigma**2))
    
    # 4. Create the modulations (cosine and sine) using Equation (3)
    cos_modulation = np.cos(2 * np.pi * freq * n)
    sin_modulation = np.sin(2 * np.pi * freq * n)
    
    # 5. Multiply Gaussian envelope with modulations to get Gabor filters
    cos_h = gaussian_env * cos_modulation
    sin_h = gaussian_env * sin_modulation
    
    return cos_h, sin_h
