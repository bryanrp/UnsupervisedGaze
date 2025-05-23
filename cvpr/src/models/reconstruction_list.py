"""Copyright 2022 Toyota Research Institute.  All rights reserved."""

from collections import OrderedDict
import numpy as np

import torch

from core.config_default import DefaultConfig
from models.reference_features import form_features_from_references

config = DefaultConfig()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReconstructionList():
    """
    Purpose: Manages feature combinations for losses.

    Key Methods:
    - add_diff_pair(): Creates pairs for feature-swapping loss.
    - add_reference_features(): Aggregates features across views. (basis loss?)

    Output: Tensors ready for generator input:
    - recon_features: (N, 152, 1, 1)  # Mixed features
    - recon_gt:       (N, 1, H, W)    # Ground truth images
    """
    def __init__(self, all_features, all_gt_frames, all_input_frames, all_valids, all_tags, unique_tags):
        self.all_features = OrderedDict(all_features) # encoder outputs
        self.all_gt_frames = all_gt_frames # augmented frames
        self.all_input_frames = all_input_frames # augmented + noised frames
        self.all_valids = all_valids # data['valids'][combo_i] = combo_tags['app'] != 'face' or not self.is_eval
        self.all_tags = all_tags # input_data['tags'] = train_data['dataset'].original_full_dataset.tags
        self.unique_tags = unique_tags # OrderedDict([('head', [0, 1]), ('gaze', [0, 1]), ('app', [0, 1]), ('sub', [0])])

        self.concat_features = []
        self.gt_frames = []
        self.reference_frames = []
        self.valid_losses = []
        self.loss_types = []

    def append(self, new_features, new_frames, new_refs, new_valids, loss_type):
        num_samples = new_features.shape[0]
        assert num_samples == new_frames.shape[0]
        assert num_samples == new_refs.shape[0]
        self.concat_features.append(new_features)
        self.gt_frames.append(new_frames)
        self.reference_frames.append(new_refs)
        self.valid_losses.append(new_valids)
        self.loss_types.extend([loss_type,] * new_features.shape[0])

    def add_diff_pair(self, name):
        if name not in self.unique_tags: # name could be 'head', 'gaze', 'app'
            return
        num_diffs = len(self.unique_tags[name])
        if num_diffs <= 1:
            # Pairwise loss requires at least 2 different types of tags
            return
        V = self.all_gt_frames.shape[1]
        C = 1
        H = self.all_gt_frames.shape[-2]
        W = self.all_gt_frames.shape[-1]

        # Loop through all tag comparisons
        for i, tag_i in enumerate(self.unique_tags[name]): # name = 'head', 'gaze', 'app';
            for tag_o in self.unique_tags[name][i+1:]:
                # Determine indices to compare
                idx_i = tag_i==self.all_tags[name]
                idx_o = tag_o==self.all_tags[name]
                # print(idx_i, idx_o, tag_i, tag_o, self.all_tags[name])
                # tensor([ True,  True, False, False,  True,  True, False, False], device='cuda:0')
                # tensor([False, False,  True,  True, False, False,  True,  True], device='cuda:0')
                # 0 1
                # tensor([0, 0, 1, 1, 0, 0, 1, 1], device='cuda:0', dtype=torch.int32)
                
                # Determine if valid pair - if not, revert to reconstruction loss
                recon_valid = torch.logical_and(self.all_valids[:,idx_i], self.all_valids[:,idx_o])
                # recon_valid.shape = torch.Size([96, 4])

                # Gather the features needed for the reconstruction loss
                recon_features = []
                for feat_name, features in self.all_features.items():
                    # feat_name = 'head', 'gaze', 'app'; features = (B,V,feat_size) = (96, 8, .)
                    # feat_size are 64, 12, 64. Total features = 140
                    recon_features.append(features[:,idx_i])
                    if feat_name != name:
                        # Swap constant features, unless invalid
                        recon_features[-1][recon_valid] = features[:,idx_o][recon_valid]
                recon_features = torch.cat(recon_features, axis=-1)
                total_features = recon_features.shape[-1]
                recon_features = recon_features.view(-1,total_features,1,1)
                recon_valid = recon_valid.view(-1)
                
                # Gather relevent frames for visualization
                recon_gt = self.all_gt_frames[:,idx_i].reshape(-1,C,H,W)
                recon_ref = self.all_input_frames[:,idx_o].reshape(-1,C,H,W)
                
                # print(recon_features.shape, recon_gt.shape, recon_ref.shape, recon_valid.shape)
                # torch.Size([384, 140, 1, 1]) torch.Size([384, 1, 128, 128]) torch.Size([384, 1, 128, 128]) torch.Size([384])
                # 384 = B * V / 2 = 96 * 8 / 2
                self.append(recon_features, recon_gt, recon_ref, recon_valid, name)

    def add_reference_features(self, reference_features):
        C = 1
        H = self.all_gt_frames.shape[-2]
        W = self.all_gt_frames.shape[-1]
        total_features = np.sum(list(config.feature_sizes.values()))

        # Assemble features
        recon_features = form_features_from_references(reference_features, self.all_tags, self.unique_tags)
        
        # Flatten everything
        total_features = recon_features.shape[-1]
        recon_features = recon_features.view(-1,total_features,1,1)
        recon_gt = self.all_gt_frames.view(-1,C,H,W)
        reconf_ref = self.all_input_frames.view(-1,C,H,W)
        recon_valid = self.all_valids.view(-1)
        self.append(recon_features, recon_gt, reconf_ref, recon_valid, 'ref')

    def join(self):
        return torch.cat(self.concat_features), torch.cat(self.gt_frames), \
               torch.cat(self.reference_frames), torch.cat(self.valid_losses), np.array(self.loss_types)
