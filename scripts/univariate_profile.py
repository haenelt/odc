# -*- coding: utf-8 -*-
"""Univariate profile."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from palettable.matplotlib import Inferno_3 as ColMap
from sklearn.feature_selection import f_classif

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import N_LAYER, SESSION, SUBJECTS
from src.data import Data, DataNew
from src.stats import Bootstrap
from src.utils import get_label

plt.style.use("src/default.mplstyle")


def select_features(dtf, label, hemi):
    # choose subset of features
    features = dtf.columns[2:]
    
    X = np.array(dtf.loc[:, features])
    y = np.array(dtf.loc[:, "label"])

    f_statistic = f_classif(X, y)[0]
    index = np.arange(len(features))
    index_sorted = np.array(
                [x for _, x in sorted(zip(f_statistic, index), reverse=True)]
            )
    index_sorted = index_sorted[: NMAX]

    label_selected = label[index_sorted]
    hemi_selected = hemi[index_sorted]

    return label_selected, hemi_selected

def layer_roi(subj, day, label, hemi, layer):
    # get features independently for each layer
    data_new = DataNew(subj, SESS, day, AREA)
    dtf = pd.read_parquet(data_new.get_sample_data_v2p0(layer))
    if NMAX:
        label_selected, hemi_selected = select_features(dtf, label, hemi)   
    else:
        label_selected = label
        hemi_selected = hemi
    return label_selected, hemi_selected


def mean_roi(subj, day, label, hemi):
    # get same features across cortical depth
    for i in range(N_LAYER):
        data_new = DataNew(subj, SESS, day, AREA)
        if i == 0:
            dtf = pd.read_parquet(data_new.get_sample_data(i))
            _batch = dtf["batch"]
            _label = dtf["label"]
            dtf = dtf.drop(columns=["batch","label"])
        else:
            _dtf = pd.read_parquet(data_new.get_sample_data(i))
            dtf = dtf + _dtf.drop(columns=["batch","label"])
    res = dtf / N_LAYER
    res.insert(0, "label", _label)
    res.insert(0, "batch", _batch)
    if NMAX:
        label_selected, hemi_selected = select_features(res, label, hemi)
    else:
        label_selected = label
        hemi_selected = hemi
    return label_selected, hemi_selected

def get_profile(sess, day):
    # get data profiles
    y = np.zeros((N_LAYER, len(SUBJECTS)))
    for i, subj in enumerate(SUBJECTS):
        label, hemi = get_label(subj)
        if sess == "VASO_uncorrected":
            data_old = Data(subj, f"VASO{SESSION[subj][sess][day]}_uncorrected")
        else:
            data_old = Data(subj, f"{sess}{SESSION[subj][sess][day]}")
        if VERSION == "v3.0":
            label_selected, hemi_selected = mean_roi(subj, day, label, hemi)

        for j in range(N_LAYER):
            if VERSION == "v2.0":
                label_selected, hemi_selected = layer_roi(subj, day, label, hemi, j)

            if METRIC == "tsnr":
                arr_left = data_old.get_tsnr("lh", j)
                arr_right = data_old.get_tsnr("rh", j)
            elif METRIC == "cnr":
                arr_left = data_old.get_cnr("lh", j)
                arr_right = data_old.get_cnr("rh", j)
            elif METRIC == "psc":
                arr_left = data_old.get_psc("lh", j)
                arr_right = data_old.get_psc("rh", j)
            elif METRIC == "rest":
                arr_left = data_old.get_rest("lh", j)
                arr_right = data_old.get_rest("rh", j)
            else:
                ValueError("Unknown metric!")

            tmp = np.concatenate((arr_left[label_selected[hemi_selected==0]], 
                                  arr_right[label_selected[hemi_selected==1]]), axis=0)
            y[j, i] = np.mean(tmp)
    return y

# univariate profile
x = np.linspace(0, 1, N_LAYER)
y1 = get_profile(SESS, 0)
y2 = get_profile(SESS, 1)
y3 = np.append(y1, y2, axis=1)
if SESS == "VASO":
    y1 *= -1
    y2 *= -1
    y3 *= -1
#np.savez(Path("/data/pt_01880") / f"{METRIC}_{SESS}.npy", y1=y1, y2=y2, y3=y3)

ci_low = []
ci_high = []
for i in range(N_LAYER):
    boot = Bootstrap(y3[i,:])
    low, high = boot.confidence_interval()
    ci_low.append(low)
    ci_high.append(high)

fig, ax = plt.subplots()
color = ColMap.hex_colors
ax.plot(x, np.mean(y1, axis=1), color=color[1], linestyle="-", label="Mean across subjects (first session)")
ax.plot(x, np.mean(y2, axis=1), color=color[1], linestyle="--", label="Mean across subjects (second session)")
ax.plot(x, np.mean(y3, axis=1), color=color[0], linestyle="-", label="Mean across subjects (both sessions)", lw=3)
ax.fill_between(x, ci_low, ci_high, color=color[0], alpha=0.2, lw=0)
ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
if SESS == "VASO":
    ax.set_ylabel("Negative % Signal Change")
else:
    ax.set_ylabel("% Signal Change")
ax.legend(loc="lower right")
dir_out = Path("/data/pt_01880")
file_out = dir_out / f"{METRIC}_{SESS}.svg" if NMAX else dir_out / f"{METRIC}_{SESS}_all.svg" 
fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")

# univariate profile (single)
y = (y1 + y2) / 2
color = ColMap.hex_colors
for i, subj in enumerate(SUBJECTS):
    fig, ax = plt.subplots()
    ax.plot(x, y[:, i], color=color[0], linestyle="-")
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    if SESS == "VASO":
        ax.set_ylabel("Negative % Signal Change")
    else:
        ax.set_ylabel("% Signal Change")
    dir_out = Path("/data/pt_01880")
    file_out = dir_out / f"{METRIC}_{SESS}_{subj}.svg" if NMAX else dir_out / f"{METRIC}_{SESS}_{subj}_all.svg" 
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")

# univariate profile (averaged across sessions)
y = (y1 + y2) / 2
ci_low = []
ci_high = []
for i in range(N_LAYER):
    boot = Bootstrap(y[i,:])
    low, high = boot.confidence_interval()
    ci_low.append(low)
    ci_high.append(high)

fig, ax = plt.subplots()
color = ColMap.hex_colors
ax.plot(x, np.mean(y, axis=1), color=color[0], linestyle="-", label="Mean across subjects (session_average)", lw=3)
ax.fill_between(x, ci_low, ci_high, color=color[0], alpha=0.2, lw=0)
ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
if SESS == "VASO":
    ax.set_ylabel("Negative % Signal Change")
else:
    ax.set_ylabel("% Signal Change")
ax.legend(loc="lower right")
dir_out = Path("/data/pt_01880")
file_out = dir_out / f"{METRIC}_{SESS}_session_average.svg" if NMAX else dir_out / f"{METRIC}_{SESS}_all_session_average.svg" 
fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")

if __name__ == "__main__":
    VERSION = "v3.0" # which decoding version (v2.0 or v3.0)
    METRIC = "psc"
    SESS = "VASO_uncorrected" # GE_EPI, SE_EPI, VASO, VASO_uncorrected
    NMAX = 200  # int | None
    AREA = "v1"