# -*- coding: utf-8 -*-
"""
Minimum distance between V1 and V3.

This notebook estimates the partial volume of V1 and V3 voxels to exclude the 
possibility that decoding performance in V3 could be influenced by activition in V1.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nibabel.freesurfer.io import read_geometry, read_label

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import SUBJECTS
from src.data import Data

plt.style.use("src/default.mplstyle")


def _load_label(subj, area, hemi):
    _data = Data(subj, None, None, area)
    _file = _data.labels[hemi][0]
    return read_label(_file, read_scalars=True)[0]


def _load_layer(subj, hemi, layer):
    _data = Data(subj, None, None, None)
    _file = _data.surfaces[hemi][layer]
    vtx, _ = read_geometry(_file)
    return vtx


def label_coordinates(subj, area, hemi, layer):
    _vtx = _load_layer(subj, hemi, layer)
    _label = _load_label(subj, area, hemi)
    return np.array(_vtx[_label])


def min_distance(arr_pts, pts):
    """Compute minimum distance between vertex array and reference point."""
    tmp = np.sqrt(np.sum((arr_pts - pts)**2, axis=1))
    return np.min(tmp)


def _main(dir_out, layer):
    res = []
    for subj in SUBJECTS:
        for hemi in ["lh", "rh"]:
            v1 = label_coordinates(subj, "v1", hemi, layer)
            v3 = label_coordinates(subj, "v3", hemi, layer)
            for v in v3:
                res.append(min_distance(v1, v))

    print(f"Mean: {np.mean(res)} mm")
    print(f"Min: {np.min(res)} mm")
    print(f"Std: {np.std(res)} mm")

    # make output directory
    Path(dir_out).mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots()
    ax.hist(res, color="gray", alpha=0.6, bins=20, edgecolor="white", linewidth=1.2)
    ax.set_xlabel(r"Distance in mm")
    ax.axvline(np.mean(res), color="black")
    ax.axvline(0.8, color="black", linestyle="dashed")
    ax.set_ylabel("Number of vertices")
    file_out = Path(dir_out) / f"distance_v1_v3_layer_{layer}_.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Euclidean distance between V1 and V3.",
    )
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument(
        "--layer",
        dest="layer",
        default=0,
        type=int,
        help="Cortical layer (0: white, 10: pial).",
    )
    args = parser.parse_args()

    # run
    _main(args.out, args.layer)
