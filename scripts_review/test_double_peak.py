# -*- coding: utf-8 -*-
"""Harting's dip test to test bimodal distribution of cortical profile."""

import os
import sys
from pathlib import Path

import diptest
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.utils import get_profile

plt.style.use("src/default.mplstyle")


def _main(dir_out, sess, area, version):
    y1 = get_profile(area, sess, 0, version)
    y2 = get_profile(area, sess, 1, version)
    y3 = np.append(y1, y2, axis=1)

    # test original sample
    dip, p_value = diptest.diptest(np.mean(y3, axis=1))

    # test bootstrapped samples
    dip_boot = []
    p_boot = []
    n_bootstrap = 1000
    n_profiles = np.shape(y3)[1]
    for _ in range(n_bootstrap):
        val = np.random.choice(np.arange(n_profiles), n_profiles, replace=True)
        arr = y3[:, val]
        arr = np.mean(arr, axis=1)
        dval, pval = diptest.diptest(arr)
        dip_boot.append(dval)
        p_boot.append(pval)

    # make output directory
    Path(dir_out).mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots()
    ax.hist(dip_boot, color="gray", alpha=0.6, bins=20, edgecolor="white", linewidth=1.2)
    ax.set_xlabel("Harting's dip statistic")
    ax.axvline(np.mean(dip), color="black")
    ax.set_ylabel("Frequency")
    file_out = Path(dir_out) / f"dip_statistic_{sess}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")

    fig, ax = plt.subplots()
    ax.hist(p_boot, color="gray", alpha=0.6, bins=20, edgecolor="white", linewidth=1.2)
    ax.set_xlabel("p-value")
    ax.axvline(np.mean(p_value), color="black")
    ax.set_ylabel("Frequency")
    file_out = Path(dir_out) / f"dip_p_value_{sess}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Statistical test for double peak.",
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
        "--version",
        dest="version",
        default="v3.0",
        type=str,
        help="Analysis version.",
    )
    args = parser.parse_args()

    # run
    _main(args.out, args.sess, args.area, args.version)
