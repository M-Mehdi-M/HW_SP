import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler 
from get_features import get_features

STUDENT_ID = "mahmoudi_mohammadmehdi_342C2"

def run_experiment(audio_train, audio_test, labels_train, labels_test, fs, 
                   filter_type, classifier_name):
    """
    Runs a single classification experiment (Task 6 sub-point).
    """
    print(f"\n--- Running: {classifier_name} + {filter_type.capitalize()} Filter ---")
    
    # 1. Extract Features
    print(f"Extracting features ({filter_type})...")
    feat_train = get_features(audio_train, fs, filter_type=filter_type)
    feat_test = get_features(audio_test, fs, filter_type=filter_type)
    
    # 2. Setup Classifier
    if classifier_name == 'KNN':
        # Task 6a, 6b: Standard KNN
        clf = KNeighborsClassifier()
        
    elif classifier_name == 'MinDist':
        # Task 6c, 6d: Optimized MinDist
        
        # IMPROVEMENT 1: Log-transform features (simulates Decibel scale)
        feat_train = np.log1p(feat_train)
        feat_test = np.log1p(feat_test)
        
        # IMPROVEMENT 2: Standard Scaling
        scaler = StandardScaler()
        feat_train = scaler.fit_transform(feat_train)
        feat_test = scaler.transform(feat_test)
        
        # IMPROVEMENT 3: Manhattan Distance
        clf = NearestCentroid(metric='manhattan')
    else:
        raise ValueError("Unknown classifier")

    # 3. Train and Predict
    clf.fit(feat_train, labels_train)
    
    pred_train = clf.predict(feat_train)
    pred_test = clf.predict(feat_test)
    
    acc_train = np.mean(pred_train == labels_train)
    acc_test = np.mean(pred_test == labels_test)
    
    print(f"Train Accuracy: {acc_train:.2%}")
    print(f"Test Accuracy:  {acc_test:.2%}") 
    
    return acc_test

def plot_results(results):
    """
    Task 7: Generates a bar chart comparison of results.
    """
    labels = list(results.keys())
    accuracies = list(results.values())
    
    plt.figure(figsize=(10, 6))
    # Colors: Blue, Green, Red, Purple
    bars = plt.bar(labels, accuracies, color=['#4c72b0', '#55a868', '#c44e52', '#8172b3'])
    
    plt.ylim(0, 1.0) # Accuracy is 0-1
    plt.ylabel('Acuratețe (Test)')
    plt.title(f'Rezultate Clasificare - Task 6 & 7\n{STUDENT_ID}')
    plt.grid(axis='y', alpha=0.3)
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2%}', ha='center', va='bottom')
    
    filename = f"{STUDENT_ID}_rezultate_clasificare.png"
    plt.savefig(filename)
    plt.close()
    print(f"\n[Task 7] Grafic salvat: {filename}")

def run_classification_suite():
    """
    Main logic to load data and run all 4 classification tasks.
    """
    print("Loading data.mat...")
    try:
        data = scipy.io.loadmat('data.mat')
    except FileNotFoundError:
        print("Error: 'data.mat' not found. Please ensure it is in the same folder.")
        return

    # Using FULL dataset
    audio_train = data['audio_train'].T
    audio_test = data['audio_test'].T
    labels_train = data['labels_train'][:, 0]
    labels_test = data['labels_test'][:, 0]
    fs = data['fs'][0, 0]

    results = {}

    # --- Task 6a: KNN + Gabor ---
    acc = run_experiment(audio_train, audio_test, labels_train, labels_test, fs,
                         filter_type='gabor', classifier_name='KNN')
    results['KNN\nGabor'] = acc

    # --- Task 6b: KNN + Custom ---
    acc = run_experiment(audio_train, audio_test, labels_train, labels_test, fs,
                         filter_type='custom', classifier_name='KNN')
    results['KNN\nCustom'] = acc

    # --- Task 6c: MinDist + Gabor ---
    acc = run_experiment(audio_train, audio_test, labels_train, labels_test, fs,
                         filter_type='gabor', classifier_name='MinDist')
    results['MinDist\nGabor'] = acc

    # --- Task 6d: MinDist + Custom ---
    acc = run_experiment(audio_train, audio_test, labels_train, labels_test, fs,
                         filter_type='custom', classifier_name='MinDist')
    results['MinDist\nCustom'] = acc
    
    # Generate Task 7 Graph
    plot_results(results)
