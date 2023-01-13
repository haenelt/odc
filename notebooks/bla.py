from pathlib import Path

import numpy as np
from column_filter.filt import Filter
from column_filter.io import load_mesh, load_mmap, load_overlay, load_roi
from nibabel.freesurfer.io import read_label

DIR_BASE = "/data/pt_01880/Experiment1_ODC"

surf_in = (
    "/data/pt_01880/Experiment1_ODC/p1/anatomy/layer/lh.layer_0"  # input surface mesh
)
dist_in = "/data/pt_01880/Experiment1_ODC/paper/dist/p1/lh.dist_layer_0.npy"  # input distance matrix
arr_in = "/data/pt_01880/Experiment1_ODC/p1/odc/results/Z/sampled/Z_all_left_right_GE_EPI3/lh.Z_all_left_right_GE_EPI3_layer_0.mgh"  # input overlay
file_out = "/data/pt_01880/test.parquet"  # output overlay

file_label1 = Path(DIR_BASE) / "p1" / "anatomy" / "label" / "lh.v1.label"
file_label2 = Path(DIR_BASE) / "p1" / "anatomy" / "label" / "lh.fov.label"
label1 = read_label(file_label1)
label2 = read_label(file_label2)
label = np.intersect1d(label1, label2)
label = np.sort(label)

surf = load_mesh(surf_in)
dist = load_mmap(dist_in)
arr = load_overlay(arr_in)["arr"]

filter_bank = Filter(surf["vtx"], surf["fac"], label, dist)
_ = filter_bank.fit(arr, file_out=file_out)
