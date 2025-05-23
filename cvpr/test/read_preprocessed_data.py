import bz2
import _pickle as cPickle
import numpy as np

def inspect_pbz2(filepath):
    with bz2.BZ2File(filepath, 'rb') as f:
        data = cPickle.load(f)
    
    print("=== PBZ2 Contents ===")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: array{value.shape} {value.dtype}")
            if value.ndim == 3:  # Image tensor
                print(f"  Min: {value.min():.2f} Max: {value.max():.2f}")
        else:
            print(f"{key}: {type(value)} {value}")

import matplotlib.pyplot as plt

def show_pbz2_frame(filepath):
    with bz2.BZ2File(filepath, 'rb') as f:
        data = cPickle.load(f)
    
    frame = data['frame'].transpose(1, 2, 0)  # CHW → HWC
    frame = (frame + 1) * 127.5  # [-1,1] → [0,255]
    
    plt.imshow(frame.astype('uint8'))
    plt.title(f"Gaze: {data.get('gaze_dir', 'N/A')}")
    plt.axis('off')
    plt.show()


def inspect_labels(root_dir, n_samples=5):
    """
    Walk `root_dir` for .pbz2 files, load the first `n_samples` of them,
    and print out type/shape/info for each of the four label fields.
    """
    import os
    import pickle

    # 1. Find all .pbz2 files
    pbz2_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith('.pbz2'):
                pbz2_paths.append(os.path.join(dirpath, fn))
    pbz2_paths.sort()
    
    # 2. Load up to n_samples entries
    for i, path in enumerate(pbz2_paths[:n_samples]):
        with bz2.BZ2File(path, 'r') as f:
            entry = pickle.load(f)
        
        print(f"\n=== Sample #{i+1}: {os.path.basename(path)} ===")
        for field in ('cam_gaze_dir', 'head_dir', 'gaze_pos', 'head_pos'):
            if field not in entry:
                print(f"  {field:<12}: <not present>")
                continue
            
            val = entry[field]
            # Print type
            typ = type(val)
            # If it's a numpy array, get its shape
            shp = getattr(val, 'shape', None)
            # Show first few elements
            snippet = None
            try:
                if hasattr(val, '__len__') and len(val) > 0:
                    # flatten and take first up to 5 elements
                    flat = val.reshape(-1)
                    snippet = flat[:5].tolist()
                else:
                    snippet = val
            except Exception:
                snippet = '<unprintable>'
            
            print(f"  {field:<12}: type={typ.__name__}, shape={shp}, snippet={snippet}")
    
    if not pbz2_paths:
        print("No .pbz2 files found under", root_dir)

if __name__ == "__main__":
    filepath = r"C:\Users\victus\Documents\git-project\capstone\UnsupervisedGaze\all-preprocessed\try-13-3\train\train01\step007_image_MIT-i2277207572\webcam_c\0\left\0.pbz2"

    inspect_pbz2(filepath)
    show_pbz2_frame(filepath)

    # rootpath = r"C:\Users\victus\Documents\git-project\capstone\UnsupervisedGaze\all-preprocessed\try-13-3\train\train01\step007_image_MIT-i2277207572\webcam_c\0\right"
    # inspect_labels(rootpath, n_samples=5)
