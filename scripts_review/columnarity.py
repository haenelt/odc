# -*- coding: utf-8 -*-
"""
Columnarity check.

This script checks for columnarity similar to the analysis done in Nasr et al. 2016.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.freesurfer.io import read_geometry
from scipy.stats import pearsonr
from sklearn.feature_selection import f_classif

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import SUBJECTS
from src.data import Data
from src.utils import get_label
from src.corrstats import independent_corr


# constants
_SESSIONS = ["GE_EPI", "SE_EPI", "VASO"]


class ContrastData:
    """Get contrast for set number of features."""

    LAYER = 0  # layer for feature selection (not data sampling)

    def __init__(self, subj, sess, day, area, version):
        self.subj = subj
        self.sess = sess
        self.day = day
        self.area = area
        self.version = version
        self.data = Data(subj, sess, day, area)

    def sort_features(self, nmax):
        """Sort label and hemi array based on features."""
        dtf = pd.read_parquet(self.data.get_sample_data(self.LAYER, self.version))
        label, hemi = get_label(self.subj, self.area)

        # choose subset of features
        features = dtf.columns[2:]
        
        X = np.array(dtf.loc[:, features])
        y = np.array(dtf.loc[:, "label"])

        f_statistic = f_classif(X, y)[0]
        index = np.arange(len(features))
        index_sorted = np.array([x for _, x in sorted(zip(f_statistic, index), reverse=True)])

        label_sorted = label[index_sorted]
        hemi_sorted = hemi[index_sorted]

        if nmax == 0:
            return label_sorted, hemi_sorted
        return label_sorted[:nmax], hemi_sorted[:nmax]

    def get_data(self, layer, nmax):
        """Get contrast."""
        _label, _hemi = self.sort_features(nmax)
        arr_left = self.data.get_contrast("lh", layer)
        arr_right = self.data.get_contrast("rh", layer)

        return np.concatenate((arr_left[_label[_hemi == 0]], arr_right[_label[_hemi == 1]]), axis=0)
    
    def get_thickness(self, nmax):
        """Get thickness data."""
        _label, _hemi = self.sort_features(nmax)
        arr_left = self.data.get_thickness("lh")
        arr_right = self.data.get_thickness("rh")

        return np.concatenate((arr_left[_label[_hemi == 0]], arr_right[_label[_hemi == 1]]), axis=0)
    
    def get_other(self, layer, nmax):
        """Get data on the same surface cortical thickness away."""
        _label, _hemi = self.sort_features(nmax)
        _thickness = self.get_thickness(nmax)

        vertices = {}
        vertices["lh"] = read_geometry(self.data.surfaces["lh"][layer])[0]
        vertices["rh"] = read_geometry(self.data.surfaces["rh"][layer])[0]

        label_other = np.zeros_like(_label)
        for i, (l, h, t) in enumerate(zip(_label, _hemi, _thickness)):
            hh = "lh" if h == 0 else "rh"
            vtx = vertices[hh]
            vtx0 = vtx[l, :]
            diff = vtx - vtx0
            distance = np.sqrt(np.sum(diff**2, axis=1))
            distance[distance <= t] = np.nan
            label_other[i] = np.nanargmin(distance)

        arr_left = self.data.get_contrast("lh", layer)
        arr_right = self.data.get_contrast("rh", layer)

        return np.concatenate((arr_left[label_other[_hemi == 0]], arr_right[label_other[_hemi == 1]]), axis=0)


def _main(dir_out, area, nmax, version):
    """Get univariate profiles with varying numbre of features for each subject and
    session compute average profile map."""

    data = {}
    data["subj"] = []
    data["sess"] = []
    data["day"] = []
    data["r_layer"] = []
    data["p_layer"] = []
    data["r_other"] = []
    data["p_other"] = []
    data["z_fisher"] = []
    data["p_fisher"] = []

    for subj in SUBJECTS:
        for sess in _SESSIONS:
            for day in [0, 1]:
                print(f"SUBJ {subj}, SESS: {sess}, DAY: {day}")
                contrast = ContrastData(subj, sess, day, area, version)
                data1 = contrast.get_data(layer=0, nmax=nmax)  # wm
                data2 = contrast.get_data(layer=10, nmax=nmax)  # pial
                data3 = contrast.get_other(layer=0, nmax=nmax)

                # correlation
                r_layer, p_layer = pearsonr(data1, data2)
                r_other, p_other = pearsonr(data1, data3)
                z_fisher, p_fisher = independent_corr(r_layer, r_other, len(data1))

                data["subj"].append(subj)
                data["sess"].append(sess)
                data["day"].append(day)
                data["r_layer"].append(r_layer)
                data["p_layer"].append(p_layer)
                data["r_other"].append(r_other)
                data["p_other"].append(p_other)
                data["z_fisher"].append(z_fisher)
                data["p_fisher"].append(p_fisher)
    
    # save to disk
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    file_out = Path(dir_out) / f"columnarity.csv"
    df = pd.DataFrame(data)
    df.to_csv(file_out, index=False)


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Make columnarity test.",
    )
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument(
        "--area",
        dest="area",
        default="v1",
        type=str,
        help="Cortical area from which features are selected (e.g. v1).",
    )
    parser.add_argument(
        "--nmax",
        dest="nmax",
        default=0,
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
    _main(args.out, args.area, args.nmax, args.version)
