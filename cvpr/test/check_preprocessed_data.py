import bz2
import _pickle as cPickle
import numpy as np
import matplotlib.pyplot as plt
import os

def inspect_pbz2(file_path):
    try:
        # Load the compressed data
        with bz2.BZ2File(file_path, 'rb') as f:
            entry = cPickle.load(f)
            
        print(f"=== Inspection of {os.path.basename(file_path)} ===")
        
        # 1. Show basic structure
        print("\n[Structure] Keys in entry:")
        print(list(entry.keys()))
        
        # 2. Check frame properties
        if 'frame' in entry:
            frame = entry['frame']
            print("\n[Frame Info]")
            print(f"Shape: {frame.shape}")
            print(f"Data type: {frame.dtype}")
            print(f"Value range: {np.min(frame)} - {np.max(frame)}")
            
            # 3. Try to display as image
            plt.figure(figsize=(10, 5))
            
            # Handle different channel dimensions
            if frame.ndim == 3 and frame.shape[2] == 3:  # Assume HWC
                plt.imshow(frame)
                plt.title("RGB Image (HWC)")
            elif frame.ndim == 3 and frame.shape[2] == 1:  # Grayscale
                plt.imshow(frame[:, :, 0], cmap='gray')
                plt.title("Grayscale (HWC)")
            elif frame.ndim == 2:  # 2D array
                plt.imshow(frame, cmap='gray')
                plt.title("2D Array")
            else:
                print(f"Unexpected shape {frame.shape} - showing first channel")
                plt.imshow(frame[:, :, 0], cmap='gray')
                
            plt.show()
            
        # 4. Check other fields
        print("\n[Metadata]")
        for key in ['gaze_dir', 'head_dir', 'cam_gaze_dir']:
            if key in entry:
                print(f"{key}: {entry[key].shape} {entry[key].dtype}")
                
    except Exception as e:
        print(f"ERROR inspecting {file_path}: {str(e)}")

# Example usage
sample_path = r"C:\Users\victus\Documents\git-project\capstone\UnsupervisedGaze\all-preprocessed\default\test\test01\step008_image_MIT-i2263021117\webcam_c\0\left"

for i in range(0, 5):
    if os.path.exists(sample_path):
        print("✅ Path is valid!")
        inspect_pbz2(f"{sample_path}\\{i}.pbz2")
    else:
        print("❌ Path does not exist!")
        print(f"Current working directory: {os.getcwd()}")
