import bz2
import _pickle as cPickle
from pprint import pprint

import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, 'src'))
import utils

def read_index_pbz2(filepath, max_depth=3, max_items=5):
    """Improved index reader that handles both raw MultiDict and file paths"""
    with bz2.BZ2File(filepath, 'rb') as f:
        data = cPickle.load(f)
    
    print("=== Index Contents ===")
    
    if not isinstance(data, utils.data_types.MultiDict):
        print("Unexpected data type:", type(data))
        return data

    # Enhanced traversal that handles both MultiDict and direct values
    def traverse(obj, path=None, depth=0, count=[0]):
        if path is None:
            path = []
            
        if depth > max_depth or count[0] >= max_items:
            return
        
        if isinstance(obj, utils.data_types.MultiDict):
            print("  " * depth + f"MultiDict (level={obj.level}):")
            for key, value in obj.data.items():
                print("  " * (depth+1) + f"Key: {key}")
                traverse(value, path + [key], depth+1, count)
        else:
            print("  " * depth + f"Value: {obj}")
            count[0] += 1
    
    traverse(data)
    
    # Count all leaf nodes (actual file paths)
    def count_leaves(obj):
        if isinstance(obj, utils.data_types.MultiDict):
            return sum(count_leaves(v) for v in obj.data.values())
        return 1
    
    total_files = count_leaves(data)
    print(f"\nTotal files indexed: {total_files:,}")
    
    # Show first few complete paths
    print("\nSample complete paths:")
    for i, key in enumerate(data.keys()):
        if i >= 5:
            break
        path = data[key]
        print(f"{key} → {path}")
    
    return data

# Example usage
index = read_index_pbz2(r"C:\Users\victus\Documents\git-project\capstone\UnsupervisedGaze\all-preprocessed\try-13-3\train\index.pbz2")

# Access example
if isinstance(index, utils.data_types.MultiDict):
    first_key = next(iter(index.keys()))
    print("\nFirst complete entry:")
    print("Key:", first_key)
    print("Path:", index[first_key])