import numpy as np

def mexican_hat_filter(size, sigma):
    """
    Creates a Mexican Hat (Ricker) wavelet filter.
    
    Parameters:
    size (int): The total size of the filter window (number of samples).
    sigma (float): The scale parameter (standard deviation), controlling the width 
                   and center frequency of the wavelet.
    
    Returns:
    h (numpy.ndarray): The Mexican Hat filter values.
    """
    # 1. Define time vector centered at 0
    # We center the filter at size/2
    t = np.arange(0, size) - (size / 2)
    
    # 2. Calculate the pre-factor A for normalization (energy normalization)
    # Formula: A = 2 / (sqrt(3 * sigma) * pi^(1/4))
    A = 2 / (np.sqrt(3 * sigma) * (np.pi ** 0.25))
    
    # 3. Calculate the Ricker wavelet formula
    # psi(t) = A * (1 - (t/sigma)^2) * exp(-t^2 / (2*sigma^2))
    t_sq = t ** 2
    sigma_sq = sigma ** 2
    
    term1 = 1 - (t_sq / sigma_sq)
    term2 = np.exp(-t_sq / (2 * sigma_sq))
    
    h = A * term1 * term2
    
    return h
