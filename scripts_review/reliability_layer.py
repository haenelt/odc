# -*- coding: utf-8 -*-
"""
Reliability across layers.

This scripts show the cortical profile of scan-to-scan repeatability. This is done to
exclude the possibility that the peak in deeper layers result from a change in SNR
across cortical depth.
"""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from palettable.matplotlib import Inferno_20 as ColMap
from scipy.stats import spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import N_LAYER, SUBJECTS
from src.data import Data
from src.stats import Bootstrap
from src.utils import get_label

plt.style.use("src/default.mplstyle")


class Reliability:
    """Compute between-session correlation of a contrast sampled at a specific layer."""

    def __init__(self, subj, sess, area, frac=0.1):
        self.subj = subj
        self.sess = sess
        self.area = area
        self.frac = frac  # fraction of data randomly selected

    def get_data(self, layer):
        label, hemi = get_label(self.subj, self.area)
        _files0 = Data(self.subj, self.sess, 0, self.area)
        _files1 = Data(self.subj, self.sess, 1, self.area)
        data1 = np.zeros_like(hemi)
        data2 = np.zeros_like(hemi)
        data1[hemi == 0] = _files0.get_contrast("lh", layer)[label[hemi == 0]]
        data1[hemi == 1] = _files0.get_contrast("rh", layer)[label[hemi == 1]]
        data2[hemi == 0] = _files1.get_contrast("lh", layer)[label[hemi == 0]]
        data2[hemi == 1] = _files1.get_contrast("rh", layer)[label[hemi == 1]]

        # get subset of data for statistics
        ndata = int(self.frac * len(data1))
        ind = np.arange(len(data1))
        random.shuffle(ind)
        sample_0 = data1[ind[:ndata]]
        sample_1 = data2[ind[:ndata]]
        
        return sample_0, sample_1

    def corr(self, layer):
        sample_0, sample_1 = self.get_data(layer)
        r, _ = spearmanr(sample_0, sample_1)  # spearman
        return r


def _main(dir_out, sess, area):
    """Compute r-value."""
    # array: (N_LAYER, N_SUBJECT)
    r_val = np.zeros((N_LAYER, len(SUBJECTS)))
    for i in range(N_LAYER):
        for j, subj in enumerate(SUBJECTS):
            rel = Reliability(subj, sess, area)
            r_val[i, j] = rel.corr(i)

    # compute confidence intervals
    ci_low = []
    ci_high = []
    for i in range(N_LAYER):
        boot = Bootstrap(r_val[i, :])
        low, high = boot.confidence_interval()
        ci_low.append(low)
        ci_high.append(high)

    fig, ax = plt.subplots()
    color = ColMap.hex_colors
    x = np.linspace(0, 1, N_LAYER)
    ax.plot(x, np.mean(r_val, axis=1), color=color[0], linestyle="-", lw=3)
    ax.fill_between(x, ci_low, ci_high, color=color[0], alpha=0.2, lw=0)
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    ax.set_ylabel("r-value")
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    file_out = Path(dir_out) / f"reliability_layer_{sess}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Plot psc across cortical depth.",
    )
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument(
        "--sess",
        dest="sess",
        type=str,
        help="Session name (GE_EPI, SE_EPI, VASO, VASO_uncorrected).",
    )
    parser.add_argument(
        "--area",
        dest="area",
        default="v1",
        type=str,
        help="Cortical area from which features are selected (e.g. v1).",
    )
    args = parser.parse_args()

    # run
    _main(args.out, args.sess, args.area)
