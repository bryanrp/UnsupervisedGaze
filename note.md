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

## Notepad

python src/train.py --data-views head 2 gaze 2 app 2 --train-view-duplicates head 1 gaze 1 app 1 --exp-name pt-sga-pair-lr-v222-d111-eve-0003 --dataset-name eve --cross-encoder-load-pretrained 0 --overwrite 1 --num-epochs 10 --feature-sizes sub 64 gaze 12 app 64 --loss-weights recon_gaze 1.0 recon_app 1.0 recon_head 1.0 --subsample-fold val 64 --batch-size 96 --patches-used left right --reference-confidence-softmax-beta 1000 --train-denoise-images 0

python src/extract_features.py --data-views head 1 gaze 1 app 2 --train-view-duplicates head 1 gaze 1 app 1 --exp-name ex-sga-pair-lr-v222-d111-eve-000 --dataset-name eve --cross-encoder-load-pretrained 1 --cross-encoder-checkpoint-folder outputs/checkpoints/pt-sga-pair-lr-v222-d111-eve-0003/checkpoints/last --overwrite 1 --gaze-feature-path outputs/features/ex-sga-pair-lr-v222-d111-eve-0003 --feature-sizes sub 64 gaze 12 app 64 --batch-size 512 --patches-used left right --reference-confidence-softmax-beta 1000

python src/estimate_gaze.py --exp-name eg-sga-pair-lr-v222-d111-eve-0003 --group-name eg-sga-pair-lr-v222-d111-eve-3_16-new --dataset-name eve --overwrite 1 --gaze-feature-path outputs/features/ex-sga-pair-lr-v222-d111-eve-0003 --feature-sizes sub 64 gaze 12 app 64 --train-data-workers 0 --subsample-fold train 100 val 64 --eval-features gaze sub --eval-target cam_gaze_dir --gaze-estimation-use-reference-features 0 --reference-confidence-softmax-beta 1000 --num-repeats 8
