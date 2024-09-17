# -*- coding: utf-8 -*-
"""Number of features."""

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from palettable.matplotlib import Inferno_20 as ColMap
from scipy.ndimage import gaussian_filter
from skimage.transform import resize

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.utils import get_nprofile

plt.style.use("src/default.mplstyle")


def _main(dir_out, sess, part, sigma, version):
    z1 = get_nprofile(sess, 0, version)
    z2 = get_nprofile(sess, 1, version)
    z3 = np.vstack((z1, z2))
    if part == 0:
        data = np.mean(z1, axis=0)
    elif part == 1:
        data = np.mean(z2, axis=0)
    elif part == 2:
        data = np.mean(z3, axis=0)
    else:
        raise ValueError("Unknown part!")

    nx, ny = 500, 250
    x, y = np.meshgrid(np.arange(ny), np.arange(nx))
    data_resized = resize(data, (nx, ny))
    data_filtered = gaussian_filter(data_resized, sigma)
    data_filtered *= 100

    fig, ax = plt.subplots()
    im = ax.imshow(data_filtered, cmap=ColMap.mpl_colormap)
    if sess == "VASO":
        cs = ax.contour(x, y, data_filtered, [56.0, 58.0], colors="black", linewidths=1.0)  # vaso
        im.set_clim([50, 60])
    elif sess == "GE_EPI":
        cs = ax.contour(x, y, data_filtered, [89.0, 91.0], colors="black", linewidths=1.0)  # ge_epi
        im.set_clim([85, 95])
    elif sess == "SE_EPI":
        cs = ax.contour(x, y, data_filtered, [72.0, 75.0], colors="black", linewidths=1.0)  # se_epi
        im.set_clim([67, 77])
    ax.clabel(cs, inline=True, fontsize=16)
    ax.set_xticks([0, 125, 250], [0.0, 0.5, 1.0])
    ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
    ax.set_ylabel("Number of vertices")
    cbar = plt.colorbar(im)
    cbar.set_label("Accuracy in %")
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    file_out = f"{Path(dir_out)}/n_features_{sess}_{part}.svg"
    fig.savefig(file_out, dpi=300, bbox_inches="tight", transparent=True, format="svg")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Decoding profiles for different feature numbers.",
    )
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
        "--sigma",
        dest="sigma",
        type=float,
        help="Size of gaussian filter.",
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
    _main(args.out, args.sess, args.part, args.sigma, args.version)
