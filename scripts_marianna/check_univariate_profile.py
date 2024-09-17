# -*- coding: utf-8 -*-
"""Check univariate profile for subject 2."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from fmri_tools.io.surf import write_label
from nibabel.freesurfer.io import read_label
from palettable.matplotlib import Inferno_3 as ColMap

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import N_LAYER, SESSION
from src.data import Data
from src.utils import get_label, layer_roi, mean_roi

plt.style.use("src/default.mplstyle")


# Constants
SUBJECT = "p2"
AREA = "v1"


def get_profile_marianna(subj, sess, day, area, nmax, version, use_marianna=True):
    """Get percent signal change across cortical depth."""
    if version not in ["v1.0", "v2.0", "v3.0"]:
        raise ValueError("Unknown version!")

    y = np.zeros(N_LAYER)
    label, hemi = get_label(subj, area)
    _data = Data(subj, sess, day, area)

    if version == "v3.0":
        label_selected, hemi_selected = mean_roi(
            subj, sess, day, area, label, hemi, nmax, version
        )

    for j in range(N_LAYER):
        if version == "v2.0" or version == "v1.0":
            label_selected, hemi_selected = layer_roi(
                subj, sess, day, area, label, hemi, j, nmax, version
            )
        arr_left = _data.get_psc("lh", j)
        arr_right = _data.get_psc("rh", j)

        dir_label = Path(os.path.dirname(__file__))
        if use_marianna:
            # use mariannas labels
            label_left = read_label(dir_label / "label_marianna/lh_max200_vertices.label")
            label_right = read_label(dir_label / "label_marianna/rh_max200_vertices.label")
        else:
            # use my labels and save as label files
            label_left = label_selected[hemi_selected == 0]
            label_right = label_selected[hemi_selected == 1]
            write_label(dir_label / f"label_daniel/lh.subj2_session_{SESSION[subj][sess][day]}_layer_{j}.label", label_selected[hemi_selected == 0])
            write_label(dir_label / f"label_daniel/rh.subj2_session_{SESSION[subj][sess][day]}_layer_{j}.label", label_selected[hemi_selected == 1])
        
        tmp = np.concatenate((arr_left[label_left], arr_right[label_right]), axis=0)
        y[j] = np.mean(tmp)
    return y


def _main(dir_out, sess, nmax, use_marianna, version):
    x = np.linspace(0, 1, N_LAYER)
    y1 = get_profile_marianna(SUBJECT, sess, 0, AREA, nmax, version, use_marianna)
    y2 = get_profile_marianna(SUBJECT, sess, 1, AREA, nmax, version, use_marianna)
    y3 = np.append(y1, y2)
    if sess == "VASO":
        y1 *= -1
        y2 *= -1
        y3 *= -1

    fig, ax = plt.subplots()
    color = ColMap.hex_colors
    ax.plot(x, y1, color=color[1], linestyle="-", label="First session")
    ax.plot(x, y2, color=color[1], linestyle="--", label="Second session")
    ax.plot(x, (y1 + y2)/2, color=color[0], linestyle="-", label="Mean across sessions", lw=3)
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    if sess == "VASO":
        ax.set_ylabel("Negative % Signal Change")
    else:
        ax.set_ylabel("% Signal Change")
    ax.legend(loc="lower right")
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    file_out = Path(dir_out) / f"{sess}_marianna.svg" if use_marianna else Path(dir_out) / f"{sess}_daniel.svg" 
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Check labels with Mariannas labels.",
    )
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument(
        "--sess",
        dest="sess",
        type=str,
        help="Session name (GE_EPI, SE_EPI, VASO, VASO_uncorrected).",
    )
    parser.add_argument(
        "--nmax",
        dest="nmax",
        default=200,
        type=int,
        help="Number of features.",
    )
    parser.add_argument(
        "--use_marianna",
        dest="use_marianna",
        action="store_true",
        help="Use Mariannas labels.",
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
    _main(args.out, args.sess, args.nmax, args.use_marianna, args.version)
