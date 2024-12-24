# -*- coding: utf-8 -*-
"""Utility functions."""

from pathlib import Path

import numpy as np
import pandas as pd
from fmri_decoder.data import SurfaceData
from sklearn.feature_selection import f_classif

from .config import DIR_BASE, N_LAYER, SESSION, SUBJECTS
from .data import Data

__all__ = [
    "get_profile",
    "get_nprofile",
    "get_univariate_profile",
    "get_label",
    "select_features",
    "layer_roi",
    "mean_roi",
]


def get_profile(area, sess, day, version):
    """Get accuracy across cortical depth."""
    y = np.zeros((N_LAYER, len(SUBJECTS)))
    for i, subj in enumerate(SUBJECTS):
        _path = (
            Path(DIR_BASE)
            / "paper"
            / version
            / "decoding"
            / subj
            / f"{sess}{SESSION[subj][sess][day]}"
        )
        if version == "v1.0":
            _file = _path / "bandpass_none" / "accuracy.csv"
        else:
            _file = _path / f"{area}_bandpass_none" / "accuracy.csv"
        data = np.genfromtxt(_file, delimiter=",")
        for j in range(N_LAYER):
            y[j, i] = np.mean(data[j, :] * 100)
    return y


def get_nprofile(sess, day, version):
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


def get_univariate_profile(sess, day, area, nmax, version):
    """Get percent signal change across cortical depth."""
    if version not in ["v1.0", "v2.0", "v3.0", "v4.0", "v5.0", "v6.0"]:
        raise ValueError("Unknown version!")

    y = np.zeros((N_LAYER, len(SUBJECTS)))
    for i, subj in enumerate(SUBJECTS):
        label, hemi = get_label(subj, area)
        _data = Data(subj, sess, day, area)

        if version in ["v3.0"]:
            label_selected, hemi_selected = mean_roi(
                subj, sess, day, area, label, hemi, nmax, version
            )

        for j in range(N_LAYER):
            if version == "v2.0" or version == "v1.0":
                label_selected, hemi_selected = layer_roi(
                    subj, sess, day, area, label, hemi, j, nmax, version
                )
            elif version == "v4.0":
                label_selected, hemi_selected = layer_roi(
                    subj, sess, day, area, label, hemi, 0, nmax, version
                )
            elif version == "v5.0":
                label_selected, hemi_selected = layer_roi(
                    subj, sess, day, area, label, hemi, 5, nmax, version
                )
            elif version == "v6.0":
                label_selected, hemi_selected = layer_roi(
                    subj, sess, day, area, label, hemi, 10, nmax, version
                )
            arr_left = _data.get_psc("lh", j)
            arr_right = _data.get_psc("rh", j)

            tmp = np.concatenate(
                (
                    arr_left[label_selected[hemi_selected == 0]],
                    arr_right[label_selected[hemi_selected == 1]],
                ),
                axis=0,
            )
            y[j, i] = np.mean(tmp)
    return y


def get_label(subj, area):
    """Get label from intersection of V1 and FOV."""
    data = Data(subj, None, None, area)
    surf_data = SurfaceData(data.surfaces, None, data.labels)

    label_left = surf_data.load_label_intersection("lh")
    label_right = surf_data.load_label_intersection("rh")

    hemi = np.zeros(len(label_left) + len(label_right))
    hemi[len(label_left) :] = 1
    label = np.append(label_left, label_right)

    return label, hemi


def select_features(dtf, label, hemi, nmax):
    """Choose subset of features."""
    features = dtf.columns[2:]

    X = np.array(dtf.loc[:, features])
    y = np.array(dtf.loc[:, "label"])

    f_statistic = f_classif(X, y)[0]
    index = np.arange(len(features))
    index_sorted = np.array(
        [x for _, x in sorted(zip(f_statistic, index), reverse=True)]
    )
    index_sorted = index_sorted[:nmax]

    label_selected = label[index_sorted]
    hemi_selected = hemi[index_sorted]

    return label_selected, hemi_selected


def layer_roi(subj, sess, day, area, label, hemi, layer, nmax, version):
    """Get features independently for each layer."""
    _data = Data(subj, sess, day, area)
    dtf = pd.read_parquet(_data.get_sample_data(layer, version))
    if nmax:
        label_selected, hemi_selected = select_features(dtf, label, hemi, nmax)
    else:
        label_selected = label
        hemi_selected = hemi
    return label_selected, hemi_selected


def mean_roi(subj, sess, day, area, label, hemi, nmax, version):
    """Get same features across cortcal depth."""
    for i in range(N_LAYER):
        _data = Data(subj, sess, day, area)
        if i == 0:
            dtf = pd.read_parquet(_data.get_sample_data(i, version))
            _batch = dtf["batch"]
            _label = dtf["label"]
            dtf = dtf.drop(columns=["batch", "label"])
        else:
            _dtf = pd.read_parquet(_data.get_sample_data(i, version))
            dtf = dtf + _dtf.drop(columns=["batch", "label"])
    res = dtf / N_LAYER
    res.insert(0, "label", _label)
    res.insert(0, "batch", _batch)
    if nmax:
        label_selected, hemi_selected = select_features(res, label, hemi, nmax)
    else:
        label_selected = label
        hemi_selected = hemi
    return label_selected, hemi_selected
