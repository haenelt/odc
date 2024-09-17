# -*- coding: utf-8 -*-
"""Test-retest reliability from V1 data."""

import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from palettable.matplotlib import Inferno_20 as ColMap
from scipy.stats import gaussian_kde, spearmanr

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.data import Data
from src.stats import permutation_test
from src.utils import get_label

plt.style.use("src/default.mplstyle")


def _main(dir_out, subj, sess, frac, n_shuffle):
    data_files1 = Data(subj, sess, 0, None)
    data_files2 = Data(subj, sess, 1, None)
    label, hemi = get_label(subj, "v1")
    data1 = np.zeros_like(hemi)
    data2 = np.zeros_like(hemi)
    data1[hemi == 0] = data_files1.get_contrast("lh", 5)[label[hemi == 0]]
    data1[hemi == 1] = data_files1.get_contrast("rh", 5)[label[hemi == 1]]
    data2[hemi == 0] = data_files2.get_contrast("lh", 5)[label[hemi == 0]]
    data2[hemi == 1] = data_files2.get_contrast("rh", 5)[label[hemi == 1]]

    # get subset of data for statistics
    ndata = int(frac * len(data1))
    ind = np.arange(len(data1))
    random.shuffle(ind)
    sample_0 = data1[ind[:ndata]]
    sample_1 = data2[ind[:ndata]]

    # linear fit
    m, b = np.polyfit(sample_0, sample_1, 1)
    x_fit = np.linspace(np.min(data1), np.max(data1), 100)
    y_fit = m * x_fit + b

    # statistics
    r, _ = spearmanr(sample_0, sample_1)  # spearman
    r_null = np.zeros(n_shuffle)
    for i in range(n_shuffle):
        random.shuffle(sample_1)
        r_null[i], _ = spearmanr(sample_0, sample_1)
    _, p_val = permutation_test(r, r_null)

    # calculate the point density
    xy = np.vstack([data1, data2])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = data1[idx], data2[idx], z[idx]

    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=z, s=15, marker="o", edgecolor="none", rasterized=True, cmap=ColMap.mpl_colormap)
    ax.plot(x_fit, y_fit, color="#F8870E", lw=5)
    ax.set_xlabel("z-score (session 1)")
    ax.set_ylabel("z-score (session 2)")
    ax.text(.99, .01, f"r={r:.2f}, p={p_val:.2f}", ha='right', va='bottom', transform=ax.transAxes)
    plt.colorbar(sc, label="Kernel density estimate in a.u.")
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    file_out = Path(dir_out) / f"{subj}_{sess}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")

    print(f"r: {r}")
    print(f"p: {p_val}")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(description="Reliability.")
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
        "--frac",
        dest="frac",
        type=float,
        help="Fraction of data used for permutation testing.",
    )
    parser.add_argument(
        "--n_shuffle",
        dest="n_shuffle",
        type=int,
        help="Number of permutation iterations.",
    )
    args = parser.parse_args()

    # run
    _main(args.out, args.subj, args.sess, args.frac, args.n_shuffle)
