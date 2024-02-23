import torch
import os
import numpy as np
import argparse
import pprint
import shutil
import logging
import os.path as osp
import laspy
from tree_learn.dataset import TreeDataset
from tree_learn.model import TreeLearn
from tree_learn.util import (build_dataloader, get_root_logger, load_checkpoint, ensemble, 
                             get_coords_within_shape, get_hull_buffer, get_hull, get_cluster_means,
                             propagate_preds, save_treewise, load_data, save_data, make_labels_consecutive, 
                             get_config, generate_tiles, assign_remaining_points_nearest_neighbor,
                             get_pointwise_preds, get_instances)

TREE_CLASS_IN_PYTORCH_DATASET = 0
NON_TREES_LABEL_IN_GROUPING = 0
NOT_ASSIGNED_LABEL_IN_GROUPING = -1
START_NUM_PREDS = 1

def colour_scale(offset, high):
    col = 255 * offset/high
    return [col, col, col]


logger = logging.getLogger('TreeLearn')

data_augmentations = {
    "jitter": False,
    "flip": False,
    "rot": False,
    "scaled": False,
    "point_jitter": False}

dataset = TreeDataset(data_root='/root/TreeLearn/data/train/random_crops/npz',
                      inner_square_edge_length=8,
                      training=True,
                      logger=logger,
                      data_augmentations=data_augmentations)

# print(dataset.__getitem__(0))
xyz, input_feat, instance_label, semantic_label, pt_offset_label, center, mask_inner, mask_off, mask_sem = dataset.__getitem__(0)

points = xyz.numpy()
# pt_offset_label = pt_offset_label.numpy()

sum_sq = torch.sqrt(torch.sum(pt_offset_label ** 2, dim=1))
# print(sum_sq)

# ONLY USE THIS FOR PEAKS
peaks = (xyz + pt_offset_label).numpy()

# Create a new LAS file
header = laspy.LasHeader(version="1.2", point_format=3)

header.offsets = [0, 0, 0]

header.scales = [0.001, 0.001, 0.001]
las = laspy.LasData(header)

# print(points[:,0])

# Set the points and additional fields
las.x = points[:, 0]
las.y = points[:, 1]
las.z = points[:, 2]

las.add_extra_dim(laspy.ExtraBytesParams(name="offset_dist", type=np.float64))
las.offset_dist = sum_sq
las.add_extra_dim(laspy.ExtraBytesParams(name="instance_label", type=np.float64))
las.instance_label = instance_label.numpy()



# Write the pointcloud with offset LAS file to disk
save_path = osp.join('/root/', 'true_offset.laz')
las.write(save_path)

# Peak LAS file
header = laspy.LasHeader(version="1.2", point_format=3)
header.offsets = [0,0,0]
header.scales = [0.001, 0.001, 0.001]
las = laspy.LasData(header)
las.x = peaks[:, 0]
las.y = peaks[:, 1]
las.z = peaks[:, 2]
save_path = osp.join('/root/', 'peaks.laz')
# las.write(save_path)

# Test instance label
print(f'Instance label: {torch.unique(instance_label).shape[0]}')

# Test offset labels
print(f'Number of peaks: {(torch.unique(torch.round((xyz + pt_offset_label)*10)/10, dim=0)).shape[0]}')

# Test semantic label
print(f'Semantic labels: {torch.unique(semantic_label)}')

# Test mask_inner, mask_off, mask_sem
print(f'Mask_inner == Mask_semenatic: {torch.all(mask_inner == mask_sem)}')

