import numpy as np
from create_filters import create_filter_bank
from create_custom_bank import create_custom_bank

def extract_windows(signal, window_size, stride):
    """
    Splits the audio signal into overlapping windows.
    
    Parameters:
    signal (np.ndarray): The input audio signal (1D).
    window_size (int): K, size of each window.
    stride (int): Step size between windows.
    
    Returns:
    windows (np.ndarray): Matrix of shape (F, K).
    """
    num_samples = len(signal)
    if num_samples < window_size:
        return np.array([]).reshape(0, window_size)
    
    # Calculate number of windows F
    # F = floor((N - K) / stride) + 1
    num_windows = (num_samples - window_size) // stride + 1
    
    # Create the matrix of windows
    windows = []
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        windows.append(signal[start:end])
        
    return np.array(windows)

def get_features(audio_train, fs, filter_type='gabor'):
    """
    Computes feature vectors for a set of audio files using Gabor or Custom filters.
    
    Parameters:
    audio_train (list): List of 1D audio signals.
    fs (int): Sampling frequency.
    filter_type (str): 'gabor' (Tasks 5b, 6a) or 'custom' (Tasks 5c, 6b).
    
    Returns:
    features (np.ndarray): Matrix of size (D x 2M).
    """
    M = 12          # Number of filter segments
    size = 1102     # Filter size K
    
    # --- 1. Generate Filters ---
    if filter_type == 'gabor':
        # Generate both Cos and Sin components
        filters_cos, filters_sin, _ = create_filter_bank(fs, M, size)
        
        # Reverse filters for convolution via dot product
        # We stack them into matrices of shape (M, K)
        W_cos = np.array([h[::-1] for h in filters_cos])
        W_sin = np.array([h[::-1] for h in filters_sin])
        
    elif filter_type == 'custom':
        # Generate Mexican Hat filters
        filters = create_custom_bank(fs, M, size)
        
        # Reverse filters
        W_custom = np.array([h[::-1] for h in filters])
        
    else:
        raise ValueError("Invalid filter_type. Use 'gabor' or 'custom'.")

    # --- 2. Process Signals ---
    stride_samples = int(0.012 * fs) # 12ms stride
    
    feature_vectors = []
    
    for signal in audio_train:
        # 5.a Create Windows (matrix F x K)
        windows = extract_windows(signal, size, stride_samples)
        
        if windows.shape[0] == 0:
            # Handle very short signals safely
            feature_vectors.append(np.zeros(2 * M))
            continue
            
        # 5.b / 5.c Apply Filtering
        if filter_type == 'gabor':
            # Apply both filters via Matrix Multiplication: (F x K) * (K x M) -> (F x M)
            # We use .T to transpose the filter matrix from (M, K) to (K, M)
            resp_cos = np.dot(windows, W_cos.T)
            resp_sin = np.dot(windows, W_sin.T)
            
            # Compute Modulus (Magnitude of the complex Gabor response)
            # sqrt(cos^2 + sin^2)
            response = np.sqrt(resp_cos**2 + resp_sin**2)
            
        else:
            # Custom filter (Mexican Hat is real-valued)
            resp = np.dot(windows, W_custom.T)
            
            # Apply Modulus (Absolute value for real filters)
            response = np.abs(resp)
            
        # Calculate Statistics (Mean and Std) -> Vector of size 2M
        # We aggregate over the time dimension (axis 0) to get one value per filter
        mean_val = np.mean(response, axis=0)
        std_val = np.std(response, axis=0)
        
        # Concatenate to form vector of size 2M
        features = np.concatenate([mean_val, std_val])
        feature_vectors.append(features)
        
    # Return matrix D x 2M
    return np.array(feature_vectors)
