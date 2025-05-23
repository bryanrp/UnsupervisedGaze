import pickle
import numpy as np
import os
from collections import defaultdict

def inspect_pkl(pkl_path, num_samples_preview=1):
    """
    Load and inspect a .pkl file produced by the feature extraction pipeline.
    
    Args:
        pkl_path (str): Path to the .pkl file.
        num_samples_preview (int): Number of samples to preview from the 'samples' list.

    Returns:
        dict: The loaded data dictionary.
    """
    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    print("=== Top-Level Inspection ===")
    print(f"Keys: {list(data.keys())}")
    for key, value in data.items():
        print(f"\nKey: '{key}'")
        print(f"  Type: {type(value)}")
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
        elif isinstance(value, dict):
            print(f"  Sub-keys: {list(value.keys())}")
        elif hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}, dtype: {value.dtype}")
        else:
            print(f"  Value: {value}")

    # Inspect samples
    samples = data.get('samples', [])
    print(f"\n=== 'samples' List ===")
    print(f"Total samples: {len(samples)}")
    for idx, sample in enumerate(samples[:num_samples_preview]):
        print(f"\n--- Sample {idx} ---")
        for s_key, s_val in sample.items():
            print(f"Field: '{s_key}' | Type: {type(s_val)}", end='')
            if isinstance(s_val, (list, tuple)):
                print(f" | Length: {len(s_val)}")
            elif isinstance(s_val, np.ndarray):
                print(f" | Shape: {s_val.shape}, dtype: {s_val.dtype}")
            elif isinstance(s_val, dict):
                subkeys = list(s_val.keys())
                print(f" | Sub-keys: {subkeys}")
                for subk in subkeys:
                    subv = s_val[subk]
                    arr = np.array(subv)
                    print(f"    - '{subk}': type {type(subv)}, shape {arr.shape}")
            else:
                print(f" | Value: {s_val}")

    return data

def preview_labels_from_pkl(pkl_path, num_entries=10):
    """
    Load a .pkl file and preview the first `num_entries` tags, sample paths, 
    and labels (gaze_dirs, head_dirs, and any 'app' tag if present).
    
    Args:
        pkl_path (str): Path to the .pkl file.
        num_entries (int): Number of entries/samples to preview.
    """
    # Load the pickle file
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    tags = data.get('tags', None)
    print("=== Top-Level Tags Preview ===")
    if tags is not None:
        # Ensure tags is sliceable: convert to list if not
        try:
            sliced_tags = tags[:num_entries]
        except Exception:
            sliced_tags = list(tags)[:num_entries]
        print(f"Total tags: {len(tags)}")
        for i, tag in enumerate(sliced_tags):
            print(f"{i:2d}: {tag}")
    else:
        print("No 'tags' key found in the data.")
    
    print("\n=== First Sample Labels Preview ===")
    samples = data.get('samples', [])
    total_samples = len(samples)
    print(f"Total samples: {total_samples}")
    
    for i, sample in enumerate(samples[:num_entries]):
        print(f"\nSample {i:2d}:")
        # Paths
        paths = sample.get('paths', None)
        if paths is not None:
            print("  paths:")
            for vp in paths:
                print(f"    - {vp}")
        # Gaze direction
        gaze = sample.get('gaze_dirs', None)
        print("  gaze_dirs:", np.array(gaze))
        # Head direction
        head = sample.get('head_dirs', None)
        print("  head_dirs:", np.array(head))
        # App tag or directions, if present
        app = sample.get('app_dirs', None) or sample.get('app', None)
        if app is not None:
            print("  app_dirs/app:", np.array(app))

def check_permutations_by_frame(pkl_path, expected_num=4):
    """
    Verify that each frame (identified by its experiment tag and frame index)
    has exactly 'expected_num' samples (one per reference camera).

    For example, we have head = 4, gaze = 1, and app = 2, so we expect 4 * 1 = 4 samples
    If we have head = 2, gaze = 2, and app = 2, we expect 2 * 2 = 4 samples
    I don't know why app is not counted, but yea it's not.
    
    Args:
        pkl_path (str): Path to the .pkl file.
        expected_num (int): Expected number of permutations per frame.
    """
    # Load data
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    samples = data.get('samples', [])
    
    # Group by (experiment_tag, frame_index)
    groups = defaultdict(list)
    for sample in samples:
        # Use the first path to extract tag and frame index
        first_path = sample['paths'][0]
        parts = first_path.split(os.sep)
        # Extract 'step...' tag folder and frame index (assumed at fixed positions from end)
        tag = parts[-5]          # e.g., 'step040_wikipedia_wikipedia-random'
        frame_idx = parts[-3]    # e.g., '176'
        key = (tag, frame_idx)
        groups[key].append(sample)
    
    # Report
    total_frames = len(groups)
    print(f"Total unique frames: {total_frames}")
    issue_count = 0
    for key, grp in groups.items():
        cnt = len(grp)
        if cnt != expected_num:
            issue_count += 1
            tag, fidx = key
            cameras = [s['paths'][0].split(os.sep)[-4:] for s in grp]
            print(f"Frame {tag}#{fidx} has {cnt} samples (expected {expected_num}). Reference cameras: {cameras}")
    if issue_count == 0:
        print("All frames have the expected number of permutations.")
    else:
        print(f"{issue_count} frame(s) with unexpected sample counts.")

if __name__ == "__main__":
    pickle_path = "outputs/features/1745289643-2_1_4/train.pkl"
    # inspect_pkl(pickle_path, num_samples_preview=3)
    # preview_labels_from_pkl(pickle_path, num_entries=5)
    check_permutations_by_frame(pickle_path, expected_num=4)
