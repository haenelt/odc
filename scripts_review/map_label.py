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
from sklearn.feature_selection import f_classif
from fmri_tools.io.surf import write_mgh

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.data import Data
from src.utils import get_label


class LabelData:
    """Get contrast for set number of features."""

    LAYER = 5  # layer for feature selection (not data sampling)

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
    
    def map(self, hemi, nmax):
        """Get data on the same surface cortical thickness away."""
        _label, _hemi = self.sort_features(nmax)
        ind_hemi = 0 if hemi == "lh" else 1
        arr = np.zeros_like(self.data.get_contrast(hemi, self.LAYER))
        arr[_label[_hemi == ind_hemi]] = 1
        
        return arr


def _main(dir_out, subj, sess, day, area, nmax, version):
    """Get label overlay for one session."""

    label_data = LabelData(subj, sess, day, area, version)
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    for hemi in ["lh", "rh"]:
        data = label_data.map(hemi, nmax)
        file_out = Path(dir_out) / f"{hemi}.label.mgh"
        write_mgh(file_out, data)


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Map label.",
    )
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument(
        "--subj",
        dest="subj",
        type=str,
        help="Subject name (p1,..., p5.).",
    )
    parser.add_argument(
        "--sess",
        dest="sess",
        type=str,
        help="Session name (GE_EPI, SE_EPI, VASO, VASO_uncorrected).",
    )
    parser.add_argument(
        "--day",
        dest="day",
        type=int,
        help="Day (0 or 1).",
    )
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
    _main(args.out, args.subj, args.sess, args.day, args.area, args.nmax, args.version)
