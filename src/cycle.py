# -*- coding: utf-8 -*-
"""Class for analysis of cycle frequency."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_LAYER, SESSION, SUBJECTS


class GetProfile:
    """
    Get profiles for 2D histogram of cycle estimations.

    Parameters
    ----------
    sess : str
        Session (GE-EPI, SE_EPI or VASO)
    filter_type : str
        Filter type used for cycle estimation (filter or scale)
    bins : int
        Number of histogram bins. The default is 20.

    Attributes
    ----------
    dir_scale : str
        Path to data for cycle estimation with scale.
    dir_filter : str
        Path to data for cycle estimation with filter.

    """

    def __init__(self, sess, filter_type, bins=20):
        self.sess = sess
        self.filter_type = filter_type
        self.bins = bins

        self.dir_scale = "/data/pt_01880/Experiment1_ODC/paper/spatial_scale"
        self.dir_filter = "/data/pt_01880/Experiment1_ODC/paper/filter_bank"

    @property
    def run(self):
        """Compute histograms"""
        y = np.zeros((N_LAYER, self.bins))
        x = np.zeros((N_LAYER, self.bins))
        for i in range(N_LAYER):
            tmp = []
            for subj in SUBJECTS:
                for day in [0, 1]:
                    data = self._get_data(subj, day, i)
                    tmp.extend(data)
            y[i, :], edges = np.histogram(tmp, self.bins)
            x[i, :] = (edges[:-1] + edges[1:]) / 2
        return x, y

    def _get_data(self, subj, day, layer):
        """Load data."""
        if self.filter_type == "scale":
            data = self._get_data_scale(subj, day, layer)
        elif self.filter_type == "filter":
            data = self._get_data_filter(subj, day, layer)
        else:
            raise ValueError("Unknown filter_type!")
        return data

    def _get_data_scale(self, subj, day, layer):
        """Load data for scale."""
        path = Path(self.dir_scale) / subj / f"{self.sess}{SESSION[subj][self.sess][day]}"
        file1 = path / f"lh.spatial_scale_layer_{layer}.npy"
        file2 = path / f"rh.spatial_scale_layer_{layer}.npy"
        data1 = np.load(file1, allow_pickle=True).flat[0]
        data2 = np.load(file2, allow_pickle=True).flat[0]
        return np.concatenate((2 * data1["length"], 2 * data2["length"]), axis=0)

    def _get_data_filter(self, subj, day, layer):
        """Load data for filter."""
        path = Path(self.dir_filter) / subj / f"{self.sess}{SESSION[subj][self.sess][day]}"
        file1 = path / f"lh.filter_bank_layer_{layer}.parquet"
        file2 = path / f"rh.filter_bank_layer_{layer}.parquet"
        data1 = pd.read_parquet(file1)
        data2 = pd.read_parquet(file2)
        return np.concatenate((data1["lambda"], data2["lambda"]), axis=0)
