# -*- coding: utf-8 -*-
"""Some helper functions."""

from pathlib import Path

import numpy as np
from fmri_tools.io.surf import write_label, write_mgh
from nibabel.freesurfer.io import read_geometry
from tqdm import tqdm

from .config import SESSION
from .data import Data


def get_composed_label(file_out, file_in, file_in2, file_ind2, file_ind3):
    """This function gets the index array of mgh files in the deformed surface mesh to
    the original dense freesurfer mesh.

    Parameters
    ----------
    file_out : str
        File name of written label file containing the composed transformation.
    file_in : str
        File name of final surface mesh (<hemi>.white_match_final).
    file_in2 : str
        File name of refined surface mesh (<hemi>.white_def2_refined).
    file_ind2 : str
        File name of label array containing the transform from refined (after GBB) to
        dense_epi (before GBB) (<hemi>.white_def2_refined_ind.txt).
    file_ind3 : str
        File name of label array containing the transform from the cutted surface mesh
        after applying the deformation to the dense freesurfer mesh
        (<hemi>.white_def2_ind).

    Returns
    -------
    None.
    """
    # make output folder
    Path(file_out).parent.mkdir(exist_ok=True, parents=True)

    vtx, _ = read_geometry(file_in)
    vtx2, _ = read_geometry(file_in2)
    ind = []
    for i in tqdm(range(len(vtx))):
        diff = np.sqrt(
            (vtx[i, 0] - vtx2[:, 0]) ** 2
            + (vtx[i, 1] - vtx2[:, 1]) ** 2
            + (vtx[i, 2] - vtx2[:, 2]) ** 2
        )
        diff_min = np.min(diff)
        ind_min = np.argmin(diff)
        if diff_min == 0:
            ind.append(ind_min)
    ind2 = np.loadtxt(file_ind2, dtype=int)
    ind3 = np.loadtxt(file_ind3, dtype=int)
    ind_final = ind3[ind2[ind]]
    write_label(file_out, ind_final)
