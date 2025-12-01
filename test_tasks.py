import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from gabor_filter import gabor_filter
from create_filters import create_filter_bank, plot_results
from mexican_hat_filter import mexican_hat_filter
from create_custom_bank import create_custom_bank, plot_custom_results
from get_features import get_features, extract_windows
from classification import run_classification_suite

STUDENT_ID = "mahmoudi_mohammadmehdi_342C2"
data = scipy.io.loadmat('data.mat')
fs_test = data['fs'][0, 0]

def test_gabor_implementation():
    print("--- Testing Task 1: Gabor Filter Implementation ---")
    size = 100
    sigma = 10.0
    freq = 0.1
    cos_h, sin_h = gabor_filter(size, sigma, freq)
    if cos_h.shape == (size,) and sin_h.shape == (size,):
        print("[PASS] Task 1 structure looks good.")
    else:
        print("[FAIL] Incorrect shape.")

def test_filter_bank_generation():
    print("\n--- Testing Task 2: Filter Bank Generation ---")
    filters_cos, filters_sin, params = create_filter_bank(fs_test, M=12, size=1102)
    ref_f = 0.00267
    ref_sigma = 187.21221
    f0, sigma0 = params[0]
    if np.isclose(f0, ref_f, atol=1e-4) and np.isclose(sigma0, ref_sigma, atol=1e-1):
        print(f"[PASS] Parameters match PDF (f={f0:.5f}, sigma={sigma0:.5f}).")
    else:
        print("[WARNING] Parameters deviate.")
    
    OUTPUT_DIR = "task_1-2"
    plot_results(filters_cos, filters_sin, fs_test, student_id=STUDENT_ID, save_dir=OUTPUT_DIR)

def test_custom_filter():
    print("\n--- Testing Task 3 & 4: Mexican Hat Filter ---")
    custom_filters = create_custom_bank(fs_test, M=12, size=1102)
    if len(custom_filters) == 12:
        print("[PASS] Created 12 custom filters.")
    else:
        print(f"[FAIL] Created {len(custom_filters)} filters.")
    
    OUTPUT_DIR = "task_3-4"
    plot_custom_results(custom_filters, fs_test, student_id=STUDENT_ID, save_dir=OUTPUT_DIR)

def test_feature_extraction():
    print("\n--- Testing Task 5: Feature Extraction (5a, 5b, 5c) ---")
    # Still use dummy audio here just to check windowing logic shapes fast
    duration_sec = 0.1
    num_samples = int(duration_sec * fs_test)
    num_files = 3
    dummy_audio = [np.random.rand(num_samples) for _ in range(num_files)]
    
    # 2. Test Window Extraction (5a)
    window_size = 1102
    stride = int(0.012 * fs_test)
    windows = extract_windows(dummy_audio[0], window_size, stride)
    expected_windows = (num_samples - window_size) // stride + 1

    if windows.shape == (expected_windows, window_size):
        print(f"[PASS] Window shape correct: {windows.shape}")
    else:
        print(f"[FAIL] Window shape mismatch.")

    # 3. Test Gabor Features (5b)
    feats_gabor = get_features(dummy_audio, fs_test, filter_type='gabor')
    if feats_gabor.shape == (num_files, 24):
        print(f"[PASS] Gabor Features shape correct: {feats_gabor.shape}")
    else:
        print(f"[FAIL] Gabor Features shape incorrect.")

    # 4. Test Custom Features (5c)
    feats_custom = get_features(dummy_audio, fs_test, filter_type='custom')
    if feats_custom.shape == (num_files, 24):
        print(f"[PASS] Custom Features shape correct: {feats_custom.shape}")
    else:
        print(f"[FAIL] Custom Features shape incorrect.")

def test_classification_pipeline():
    print("\n--- Testing Task 6 & 7: Classification on REAL DATA ---")

    run_classification_suite()
    
    print("\n[PASS] Classification suite finished.")

if __name__ == "__main__":
    test_gabor_implementation()
    test_filter_bank_generation()
    test_custom_filter()
    test_feature_extraction()
    test_classification_pipeline()
    print("\nAll tests completed successfully.")
