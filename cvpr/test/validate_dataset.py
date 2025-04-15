import os
import bz2
import _pickle as cPickle

import pathlib
import sys
file_dir_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(file_dir_path, ".."))
from utils.data_types import MultiDict

def get_all_paths(multi_dict):
    """Extract paths from MultiDict matching the original structure"""
    paths = []
    # Iterate through camera types (basler/webcam_l/etc)
    for camera in multi_dict.data.keys():
        camera_dict = multi_dict[camera]
        # Iterate through participants
        for participant in camera_dict.data.keys():
            participant_dict = camera_dict[participant]
            # Iterate through stimulus numbers
            for stimulus_num in participant_dict.data.keys():
                stimulus_dict = participant_dict[stimulus_num]
                # Iterate through patch types (left/right/face)
                for patch_type in stimulus_dict.data.keys():
                    patch_dict = stimulus_dict[patch_type]
                    # Iterate through time steps
                    for time_step in patch_dict.data.keys():
                        paths.append(
                            os.path.join(
                                participant,
                                camera,
                                str(stimulus_num),
                                patch_type,
                                f"{time_step}.pbz2"
                            )
                        )
    return paths

def validate_dataset(dataset_path):
    index_path = os.path.join(dataset_path, 'index.pbz2')
    with bz2.BZ2File(index_path, 'rb') as f:
        patches = cPickle.load(f)

    for key, multi_dict in patches.items():
        print(multi_dict)
        rel_paths = get_all_paths(multi_dict)
        
        for rel_path in rel_paths:
            full_path = os.path.join(dataset_path, rel_path)
            
            try:
                with bz2.BZ2File(full_path, 'rb') as f:
                    entry = cPickle.load(f)
                    frame = entry['frame']
                    assert frame.shape == (256, 256, 3), \
                        f"Bad shape in {rel_path}: {frame.shape}"
                    print(f"Good {full_path}")
            except Exception as e:
                print(f"Failed to validate {rel_path}: {str(e)}")
                raise

validate_dataset(r"C:\Users\victus\Documents\git-project\capstone\UnsupervisedGaze\all-preprocessed\rgb-24\train")
# validate_dataset(r"C:\Users\victus\Documents\git-project\capstone\UnsupervisedGaze\all-preprocessed\rgb-24\test")
# validate_dataset(r"C:\Users\victus\Documents\git-project\capstone\UnsupervisedGaze\all-preprocessed\rgb-24\val")
