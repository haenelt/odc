# -*- coding: utf-8 -*-
"""Decoding profile."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from palettable.matplotlib import Inferno_3 as ColMap

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import N_LAYER, SUBJECTS
from src.stats import Bootstrap, fdr_correction
from src.utils import get_profile

plt.style.use("src/default.mplstyle")


def _main(dir_out, sess, area, version):
    x = np.linspace(0, 1, N_LAYER)
    y1 = get_profile(area, sess, 0, version)
    y2 = get_profile(area, sess, 1, version)
    y3 = np.append(y1, y2, axis=1)

    ci_low = []
    ci_high = []
    p_adj = []
    for i in range(N_LAYER):
        boot = Bootstrap(y3[i, :])
        low, high = boot.confidence_interval()
        ci_low.append(low)
        ci_high.append(high)
        p_adj.append(boot.p_value())
    p_adj = fdr_correction(p_adj)

    # make output directory
    Path(dir_out).mkdir(exist_ok=True, parents=True)

    # decoding profile
    fig, ax = plt.subplots()
    color = ColMap.hex_colors
    ax.plot(x, np.mean(y1, axis=1), color=color[1], linestyle="-", label="Mean across subjects (first session)")
    ax.plot(x, np.mean(y2, axis=1), color=color[1], linestyle="--", label="Mean across subjects (second session)")
    ax.plot(x, np.mean(y3, axis=1), color=color[0], linestyle="-", label="Mean across subjects (both sessions)", lw=3)
    #ax.scatter(x[p_adj[i]<0.05],np.mean(y3, axis=1)[p_adj[i]<0.05], color=color[0], s=100)
    ax.fill_between(x, ci_low, ci_high, color=color[0], alpha=0.2, lw=0)
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    ax.set_ylabel("Accuracy in %")
    ax.legend(loc="lower right")
    file_out = Path(dir_out) / f"decoding_{sess}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")

    # decoding profile of single subjects
    y = (y1 + y2) / 2
    color = ColMap.hex_colors
    for i, subj in enumerate(SUBJECTS):
        fig, ax = plt.subplots()
        ax.plot(x, y[:, i], color=color[0], linestyle="-")
        ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
        ax.set_ylabel("Accuracy in %")
        file_out = Path(dir_out) / f"decoding_{sess}_{subj}.svg"
        fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")

    # decoding profile (averaged across sessions)
    y = (y1 + y2) / 2
    ci_low = []
    ci_high = []
    p_adj = []
    for i in range(N_LAYER):
        boot = Bootstrap(y[i,:])
        low, high = boot.confidence_interval()
        ci_low.append(low)
        ci_high.append(high)
        p_adj.append(boot.p_value())
    p_adj = fdr_correction(p_adj)

    fig, ax = plt.subplots()
    color = ColMap.hex_colors
    ax.plot(x, np.mean(y, axis=1), color=color[0], linestyle="-", label="Mean across subjects (session average)", lw=3)
    ax.fill_between(x, ci_low, ci_high, color=color[0], alpha=0.2, lw=0)
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    ax.set_ylabel("Accuracy in %")
    ax.legend(loc="lower right")
    file_out = Path(dir_out) / f"decoding_{sess}_session_average.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(description="Decoding profile.")
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
