## Read src/README.md

### TODO 
- They use conda. find a way to use venv
- Preprocess EVE dataset. find a way to use ETH-XGaze
- There are 3 steps in running.
    - Train the unsupervised feature representation
    - Extract feature representation to gaze direction
    - Last, combine both to get the final gaze direction alg
- We can run all above through `./run_experiments.py --exp-txt exps/example.txt`

### I changed

- src/preprocessing/common.py: predefined_eve_splits. i make it less train val test
- src/core/config_default.py: eve_raw_path and eve_preprocessed_path
- src/preprocessing/eve.py: add logging on process_all_stimuli and load_all_from_source function. also change in get_frames()
- i downloaded ffmpeg and put it in C:/ffmpeg. its a binary files, so we can js remove it. dont forget to remove it from Path environment variables
- 
