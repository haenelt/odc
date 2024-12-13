# -*- coding: utf-8 -*-
"""
Univariate profile with varying number of features.

This scripts shows the percent signal changes for the best 200 verticies across cortial
depth for a varying number of feeatures (vertices). This is done to have an euqivalent
figure to the decoding profile with varying number of features.
"""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from palettable.matplotlib import Inferno_20 as ColMap
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
from sklearn.feature_selection import f_classif

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import N_LAYER, SUBJECTS
from src.data import Data
from src.utils import get_label

plt.style.use("src/default.mplstyle")


class Univariate:
    """Compute the univariate profile for different number of features."""

    def __init__(self, subj, sess, day, area, version):
        self.subj = subj
        self.sess = sess
        self.day = day
        self.area = area
        self.version = version
        self.data = Data(subj, sess, day, area)
        self.label, self.hemi = get_label(subj, area)
        self.label_sorted, self.hemi_sorted = zip(*[self.sort_features(i) for i in range(N_LAYER)])

    def sort_features(self, layer):
        """Sort label and hemi array based on features."""
        dtf = pd.read_parquet(self.data.get_sample_data(layer, self.version))

        # choose subset of features
        features = dtf.columns[2:]
        
        X = np.array(dtf.loc[:, features])
        y = np.array(dtf.loc[:, "label"])

        f_statistic = f_classif(X, y)[0]
        index = np.arange(len(features))
        index_sorted = np.array([x for _, x in sorted(zip(f_statistic, index), reverse=True)])

        label_sorted = self.label[index_sorted]
        hemi_sorted = self.hemi[index_sorted]

        return label_sorted, hemi_sorted

    def get_profile(self, nmax):
        """Get data profile."""
        y = np.zeros(N_LAYER)
        for i in range(N_LAYER):
            label = self.label_sorted[i][:nmax]
            hemi = self.hemi_sorted[i][:nmax]

            # load psc
            arr_left = self.data.get_psc("lh", i)
            arr_right = self.data.get_psc("rh", i)

            tmp = np.concatenate((arr_left[label[hemi == 0]], arr_right[label[hemi == 1]]), axis=0)
            y[i] = np.mean(tmp)
        return y
    
    def n_features(self, n):
        """Get data profiles for varying number of features n."""
        y = np.zeros((n, N_LAYER))
        for i in range(n):
            y[i, :] = self.get_profile(i + 1)
        if self.sess == "VASO":
            y *= -1
        return y


def _main(dir_out, sess, area, sigma, nmax, version):
    """Get univariate profiles with varying numbre of features for each subject and
    session compute average profile map."""
    counter = 0
    data = np.zeros((nmax, N_LAYER))
    for subj in SUBJECTS:
        for day in [0, 1]:
            data += Univariate(subj, sess, day, area, version).n_features(nmax)
            counter += 1
    data /= counter

    # regrid map and smooth
    nx, ny = 500, 250
    x, y = np.meshgrid(np.arange(ny), np.arange(nx))
    data_resized = resize(data, (nx, ny))
    data_filtered = gaussian_filter(data_resized, sigma)

    if sess == "VASO":
        contour_range = [0.7, 0.9]
        clim_range = [0, 1.0]
    elif sess == "GE_EPI":
        contour_range = [2.4, 3.4]
        clim_range = [0, 5]
    elif sess == "SE_EPI":
        contour_range = [2.0, 2.6]
        clim_range = [0, 3.0]
    elif sess == "VASO_uncorrected":
        contour_range = [2.4, 3.4]
        clim_range = [0, 5]

    fig, ax = plt.subplots()
    im = ax.imshow(data_filtered, cmap=ColMap.mpl_colormap)
    cs = ax.contour(x, y, data_filtered, contour_range, colors="black", linewidths=1.0)
    im.set_clim(clim_range)
    ax.clabel(cs, inline=True, fontsize=16)
    ax.set_xticks([0, 125, 250], [0.0, 0.5, 1.0])
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    ax.set_ylabel("Number of vertices")
    cbar = plt.colorbar(im)
    cbar.set_label("% Signal Change")
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    file_out = Path(dir_out) / f"n_features_univariate_{sess}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Plot psc across  cortical depth.",
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
    parser.add_argument(
        "--sigma",
        dest="sigma",
        default=1.0,
        type=float,
        help="Size of gaussian filter.",
    )
    parser.add_argument(
        "--nmax",
        dest="nmax",
        default=500,
        type=int,
        help="Number of features.",
    )
    parser.add_argument(
        "--version",
        dest="version",
        default="v3.0",
        type=str,
        help="Analysis version.",
    )
    args = parser.parse_args()

    # run
    _main(args.out, args.sess, args.area, args.sigma, args.nmax, args.version)
