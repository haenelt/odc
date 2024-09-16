# -*- coding: utf-8 -*-
"""Decoding extrastriate."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from palettable.matplotlib import Inferno_3 as ColMap

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import N_LAYER
from src.stats import Bootstrap, fdr_correction
from src.utils import get_profile

plt.style.use("src/default.mplstyle")


def _main(dir_out, sess, part, area1, area2, version):
    x = np.linspace(0, 1, N_LAYER)
    data = []
    p_adj = []
    areas = [area1, area2]
    for area in areas:
        y1 = get_profile(version, area, sess, 0)
        y2 = get_profile(version, area, sess, 1)
        y3 = np.append(y1, y2, axis=1)
        if part == 0:
            data.append(np.mean(y1, axis=1))
        elif part == 1:
            data.append(np.mean(y2, axis=1))
        elif part == 2:
            data.append(np.mean(y3, axis=1))
        else:
            raise ValueError("Unknown part!")

        # get fdr-corrected p-values
        p_val = []
        for i in range(N_LAYER):
            if part == 0:
                boot = Bootstrap(y1[i, :])
            elif part == 1:
                boot = Bootstrap(y2[i, :])
            elif part == 2:
                boot = Bootstrap(y3[i, :])
            else:
                raise ValueError("Unknown error!")

            p_val.append(boot.p_value())

        p_val = fdr_correction(p_val)
        p_adj.append(p_val)

    fig, ax = plt.subplots()
    color = ColMap.hex_colors
    for i, area in enumerate(areas):
        ax.plot(x, data[i], color=color[i], linestyle="-", label=area, lw=3)
        ax.scatter(x[p_adj[i]<0.05], data[i][p_adj[i]<0.05], color=color[i], s=100)
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    ax.set_ylabel("Accuracy in %")
    ax.legend(loc="lower right")
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    file_out = Path(dir_out) / f"decoding_{sess}_{part}_{areas[0][:2]}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(description="Extrastriate decoding profile.")
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument(
        "--sess",
        dest="sess",
        type=str,
        help="Session name (GE_EPI, SE_EPI, VASO, VASO_uncorrected).",
    )
    parser.add_argument(
        "--part",
        dest="part",
        type=int,
        help="Session day (0, 1 or 2). 2 is the average across sessions.",
    )
    parser.add_argument(
        "--area1",
        dest="area1",
        type=str,
        help="First cortical area from which features are selected (e.g. v2a).",
    )
    parser.add_argument(
        "--area2",
        dest="area2",
        type=str,
        help="Second cortical area from which features are selected (e.g. v2b).",
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
    _main(args.out, args.sess, args.part, args.area1, args.area2, args.version)
