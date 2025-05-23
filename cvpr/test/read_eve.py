import h5py

def inspect_h5(file_path, samples=5):
    """
    Inspect the structure of an HDF5 file and print the first 5 samples of each dataset.
    
    Parameters:
    - file_path: Path to the .h5 file
    """
    with h5py.File(file_path, 'r') as f:
        print("=== HDF5 File Structure ===")
        
        # Recursively print groups and datasets
        def print_item(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}/")
            elif isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(print_item)
        
        # Print first samples for each dataset
        print(f"\n=== First {samples} Samples per Dataset ===")
        def print_samples(name, obj):
            if isinstance(obj, h5py.Dataset):
                # Read up to #samples elements along the first dimension
                try:
                    sample = obj[:samples]
                except Exception:
                    sample = obj[...].flatten()[:samples]
                print(f"\n{name} (first {samples} elements):\n{sample}")
        f.visititems(print_samples)

if __name__ == "__main__":
    # h5_path = r"C:\Users\victus\Downloads\eve_dataset\eve_dataset\train01\step008_image_MIT-i1677421979\webcam_c.h5"
    h5_path = r"C:\Users\victus\Downloads\eve_dataset\eve_dataset\val01\step008_image_MIT-i1086742403\webcam_c.h5"
    # h5_path = r"C:\Users\victus\Downloads\eve_dataset\eve_dataset\test01\step009_image_MIT-i2267703789\webcam_c.h5"
    inspect_h5(h5_path, 1)
