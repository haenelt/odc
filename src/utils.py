# -*- coding: utf-8 -*-
"""Utility functions."""

from pathlib import Path

import numpy as np
from fmri_decoder.data import SurfaceData

from .config import DIR_BASE, N_LAYER, SESSION, SUBJECTS
from .data import Data

__all__ = ["get_profile", "get_nprofile", "get_label"]


def get_profile(version, area, sess, day):
    """Get accuracy across cortical depth."""
    y = np.zeros((N_LAYER, len(SUBJECTS)))
    for i, subj in enumerate(SUBJECTS):
        _path = (
            Path(DIR_BASE)
            / "paper"
            / version
            / "decoding"
            / subj
            / f"{sess}{SESSION[subj][sess][day]}",
        )
        if version == "v1.0":
            _file = _path / "bandpass_none" / "accuracy.csv"
        else:
            _file = _path / f"{area}_bandpass_none" / "accuracy.csv"
        data = np.genfromtxt(_file, delimiter=",")
        for j in range(N_LAYER):
            y[j, i] = np.mean(data[j, :] * 100)
    return y


def get_nprofile(version, sess, day):
    """Get accuracy across cortical depth and across number of features."""
    data = []
    for subj in SUBJECTS:
        _path = (
            Path(DIR_BASE)
            / "paper"
            / version
            / "n_features"
            / subj
            / f"{sess}{SESSION[subj][sess][day]}"
        )
        _file = _path / "accuracy.csv"
        data.append(np.genfromtxt(_file, delimiter=","))
    return data


def get_label(subj):
    """Get label from intersection of V1 and FOV."""
    # get label and hemisphere
    data = Data(subj, None, None, "v1")
    surf_data = SurfaceData(data.surfaces, None, data.labels)

    label_left = surf_data.load_label_intersection("lh")
    label_right = surf_data.load_label_intersection("rh")

    hemi = np.zeros(len(label_left) + len(label_right))
    hemi[len(label_left):] = 1
    label = np.append(label_left, label_right)

    return label, hemi
