# -*- coding: utf-8 -*-
"""Some helper functions."""

from pathlib import Path

import numpy as np
from fmri_tools.io.surf import write_mgh

from .config import SESSION
from .data import Data


def get_average_map(file_out, subj, seq, hemi, layer):
    res = []
    for sess in SESSION[subj][seq]:
        data = Data(subj, f"{seq}{sess}")
        arr = data.get_contrast(hemi, layer)
        res.append(arr)
    res = np.mean(res, axis=0)
    dir_out = Path(file_out).parent
    dir_out.mkdir(exist_ok=True, parents=True)
    write_mgh(file_out, res)


if __name__ == "__main__":
    subj = "p1"
    seq = "VASO"
    hemi = "lh"
    layer = 5
    file_out = "/data/pt_01880/z_vaso.mgh"

    get_average_map(file_out, subj, seq, hemi, layer)
