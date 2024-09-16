# -*- coding: utf-8 -*-
"""Create maps in dense freesurfer space."""

import os
import sys
from pathlib import Path

import numpy as np
from fmri_tools.io.surf import (
    curv_to_patch,
    label_to_patch,
    mgh_to_patch,
    patch_as_mesh,
    read_mgh,
    read_patch,
    write_label,
    write_mgh,
)
from fmri_tools.surface.filter import LaplacianGaussian
from fmri_tools.surface.mesh import Mesh
from nibabel.freesurfer.io import read_geometry, read_label

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.config import DIR_BASE, SESSION
from src.helper import get_composed_label


def _main(dir_out, subj, hemi, layer):
    # make output folders
    Path(dir_out).mkdir(exist_ok=True, parents=True)
    DIR_DENSE = Path(dir_out) / subj / "dense"
    DIR_FLAT = Path(dir_out) / subj / "flat"
    DIR_DENSE.mkdir(parents=True, exist_ok=True)
    DIR_FLAT.mkdir(parents=True, exist_ok=True)

    # get final transformation label to dense freesurfer mesh
    file_out = DIR_DENSE / f"{hemi}.composed.label"
    file_in = Path(DIR_BASE) / subj / "anatomy/dense_refined" / f"{hemi}.white_match_final"
    file_in2 = Path(DIR_BASE) / subj / "anatomy/gbb" / f"{hemi}.white_def2_refined"
    file_ind2 = Path(DIR_BASE) / subj / "anatomy/gbb" / f"{hemi}.white_def2_refined_ind.txt"
    file_ind3 = Path(DIR_BASE) / subj / "anatomy/dense_epi" / f"{hemi}.white_def2_ind"
    get_composed_label(file_out, file_in, file_in2, file_ind2, file_ind3)

    # transform arrays
    file_ind = DIR_DENSE / f"{hemi}.composed.label"
    file_geom = Path(DIR_BASE) / subj / "anatomy/dense" / f"{hemi}.white"
    file_mgh = []
    for sess in SESSION[subj]:
        for day in SESSION[subj][sess]:
            if "_uncorrected" not in sess:
                file_mgh.append(
                    Path(DIR_BASE)
                    / subj 
                    / f"odc/results/Z/sampled/Z_all_left_right_{sess}{day}" 
                    / f"{hemi}.Z_all_left_right_{sess}{day}_layer_{layer}.mgh"
                )

    # also include retinotopy data
    ret_session = "retinotopy2" if subj == "p4" else "retinotopy"
    for ret_type in ["ecc", "pol"]:
        file_mgh.append(
            Path(DIR_BASE)
            / subj
            / f"{ret_session}"
            / "avg"
            / "sampled"
            / f"{ret_type}_phase_avg"
            / f"{hemi}.{ret_type}_phase_avg_layer_{layer}.mgh"
        )

    ind = read_label(file_ind)
    vtx, _ = read_geometry(file_geom)
    for f in file_mgh:
        file_out = DIR_DENSE / f"{Path(f).stem}_dense.mgh"
        arr, _, _ = read_mgh(f)
        res = np.zeros(len(vtx))
        res[ind] = arr
        write_mgh(file_out, res)

    # transform label
    file_ind = DIR_DENSE / f"{hemi}.composed.label"
    file_label = []
    for label in ["fov", "v1", "v2", "v2a", "v2b", "v3", "v3a", "v3b"]:
        file_label.append(
            Path(DIR_BASE)
            / subj 
            / "anatomy/label"
            / f"{hemi}.{label}.label",
        )

    ind = read_label(file_ind)
    for f in file_label:
        file_out = DIR_DENSE / f"{Path(f).stem}_dense.label"
        l_ = read_label(f)
        res = ind[l_]
        write_label(file_out, res)

    # get average maps
    for sess in SESSION[subj]:
        days = SESSION[subj][sess]
        if "_uncorrected" not in sess:
            file_mgh1 = (
                DIR_DENSE
                / f"{hemi}.Z_all_left_right_{sess}{days[0]}_layer_{layer}_dense.mgh"
            )
            file_mgh2 = (
                DIR_DENSE
                / f"{hemi}.Z_all_left_right_{sess}{days[1]}_layer_{layer}_dense.mgh"
            )
            file_out = DIR_DENSE / f"{hemi}.Z_all_left_right_{sess}_layer_{layer}_avg.mgh"
            arr1, _, _ = read_mgh(file_mgh1)
            arr2, _, _ = read_mgh(file_mgh2)
            res = (arr1 + arr2) / 2
            mask = np.ones_like(arr1)
            mask[arr1 == 0] = 0
            mask[arr2 == 0] = 0
            res[mask == 0] = 0
            write_mgh(file_out, res)

    #  get banpass filtered contrast
    # file_geom = Path(DIR_BASE) / subj / "anatomy/dense" / f"{hemi}.white"
    # file_label = DIR_DENSE / f"{hemi}.v1_dense.label"
    # file_mgh = DIR_DENSE / f"{hemi}.Z_all_left_right_GE_EPI_layer_{LAYER}_avg.mgh"
    # vtx, fac = read_geometry(file_geom)
    # label = read_label(file_label)
    # arr, _, _ = read_mgh(file_mgh)
    # mesh = Mesh(vtx, fac)
    # surf_roi = mesh.remove_vertices(label)
    # verts = surf_roi[0]
    # faces = surf_roi[1]
    # for filter_size in [0.025, 0.1, 2.0]:
    #     filt = LaplacianGaussian(verts, faces, filter_size)
    #     tmp = filt.apply(arr[label])
    #     res = np.zeros(len(vtx))
    #     res[label] = tmp
    #     file_out = DIR_DENSE / f"{Path(file_mgh).stem}_bandpass_{filter_size}.mgh"
    #     write_mgh(file_out, res)

    # project to patch
    file_out = DIR_FLAT / f"{hemi}.flat"
    file_patch = Path(DIR_BASE) / subj / f"anatomy/flat/{hemi}.flat.patch.flat"
    patch_as_mesh(file_out, file_patch)

    file_mgh = DIR_DENSE.glob(f"{hemi}*.mgh")
    for f_in in file_mgh:
        f_out = DIR_FLAT / f_in.name
        mgh_to_patch(f_out, f_in, file_patch)

    file_out = DIR_FLAT / f"{hemi}.curv"
    file_in = Path(DIR_BASE) / subj / "anatomy/dense" / f"{hemi}.curv"
    curv_to_patch(file_out, file_in, file_patch)

    # remove indices from label that are not within the flattened patch
    _, _, _, ind = read_patch(file_patch)
    label = ["fov", "v1", "v2", "v2a", "v2b", "v3", "v3a", "v3b"]
    for l_ in label:
        f_out = DIR_FLAT / f"{hemi}.{l_}_flat.label"
        f_in = DIR_DENSE / f"{hemi}.{l_}_dense.label"
        f_tmp = DIR_DENSE / f"{hemi}.tmp.label"
        tmp = read_label(f_in)
        tmp = np.intersect1d(tmp, ind)
        write_label(f_tmp, tmp)
        label_to_patch(f_out, f_tmp, file_patch)
        os.remove(f_tmp)


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(description="Make maps.")
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument(
        "--subj",
        dest="subj",
        type=str,
        help="Subject name (p1,..., p5.).",
    )
    parser.add_argument(
        "--hemi",
        dest="hemi",
        type=str,
        help="Cortical hemisphere (lh, rh).",
    )
    parser.add_argument(
        "--layer",
        dest="layer",
        default=5,
        type=int,
        help="Cortical layer.",
    )
    args = parser.parse_args()

    # run
    _main(args.out, args.subj, args.hemi, args.layer)