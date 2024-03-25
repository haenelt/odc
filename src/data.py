# -*- coding: utf-8 -*-
"""Filepaths to data."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fmri_tools.io.surf import read_mgh, write_mgh
from nibabel.freesurfer.io import read_geometry

from .config import DIR_BASE, DIR_CACHE, N_LAYER

__all__ = ["Data", "DataFiles"]


@dataclass
class Data:
    """Class for loading of sampled data from the decoding analysis."""

    subj: str
    session: str
    DIR_BASE = Path(DIR_BASE)

    @property
    def file_layer(self):
        """Load layer files."""
        file_ = {}
        file_["lh"] = [
            str(self.DIR_BASE / self.subj / "anatomy" / "layer" / f"lh.layer_{i}")
            for i in range(N_LAYER)
        ]
        file_["rh"] = [
            str(self.DIR_BASE / self.subj / "anatomy" / "layer" / f"rh.layer_{i}")
            for i in range(N_LAYER)
        ]
        return file_

    @property
    def file_label(self):
        """Load label files."""
        file_ = {}
        file_["lh"] = [
            str(self.DIR_BASE / self.subj / "anatomy" / "label" / "lh.fov.label"),
            str(self.DIR_BASE / self.subj / "anatomy" / "label" / "lh.v1.label"),
        ]
        file_["rh"] = [
            str(self.DIR_BASE / self.subj / "anatomy" / "label" / "rh.fov.label"),
            str(self.DIR_BASE / self.subj / "anatomy" / "label" / "rh.v1.label"),
        ]
        return file_

    def get_sample_data(self, layer):
        """Load sample data from MVPA analysis."""
        file = (
            self.DIR_BASE
            / "paper"
            / "v2.0"
            / "decoding"
            / self.subj
            / self.session
            / "v1_bandpass_none"
            / "sample"
            / f"sample_data_{layer}.parquet"
        )
        return file

    def get_contrast(self, hemi, layer):
        """Load z-contrast left > right."""
        file = (
            self.DIR_BASE
            / self.subj
            / "odc"
            / "results"
            / "Z"
            / "sampled"
            / f"Z_all_left_right_{self.session}"
            / f"{hemi}.Z_all_left_right_{self.session}_layer_{layer}.mgh"
        )
        return read_mgh(file)[0]

    def get_tsnr(self, hemi, layer):
        """Load tsnr map."""
        file = (
            self.DIR_BASE
            / self.subj
            / "odc"
            / "results"
            / "tsnr"
            / "sampled"
            / f"tsnr_all_{self.session}"
            / f"{hemi}.tsnr_all_{self.session}_layer_{layer}.mgh"
        )
        return read_mgh(file)[0]

    def get_psc(self, hemi, layer):
        """Load percent signal change averaged over left > rest and right > rest."""
        file_left = (
            self.DIR_BASE
            / self.subj
            / "odc"
            / "results"
            / "raw"
            / "sampled"
            / f"psc_all_left_rest_{self.session}"
            / f"{hemi}.psc_all_left_rest_{self.session}_layer_{layer}.mgh"
        )
        file_right = (
            self.DIR_BASE
            / self.subj
            / "odc"
            / "results"
            / "raw"
            / "sampled"
            / f"psc_all_right_rest_{self.session}"
            / f"{hemi}.psc_all_right_rest_{self.session}_layer_{layer}.mgh"
        )
        return (read_mgh(file_left)[0] + read_mgh(file_right)[0]) / 2

    def get_cnr(self, hemi, layer):
        """Load contrast-to-noise ratio averaged over left > rest and right > rest."""
        file_left = (
            self.DIR_BASE
            / self.subj
            / "odc"
            / "results"
            / "cnr"
            / "sampled"
            / f"cnr_all_left_rest_{self.session}"
            / f"{hemi}.cnr_all_left_rest_{self.session}_layer_{layer}.mgh"
        )
        file_right = (
            self.DIR_BASE
            / self.subj
            / "odc"
            / "results"
            / "cnr"
            / "sampled"
            / f"cnr_all_right_rest_{self.session}"
            / f"{hemi}.cnr_all_right_rest_{self.session}_layer_{layer}.mgh"
        )
        return (read_mgh(file_left)[0] + read_mgh(file_right)[0]) / 2

    def get_rest(self, hemi, layer):
        """Load resting-state data."""
        name_rest = "resting_state2" if self.subj == "p4" else "resting_state"
        file = (
            self.DIR_BASE
            / self.subj
            / name_rest
            / "ssw"
            / "sampled"
            / "nssw"
            / f"{hemi}.nssw_layer_{layer}.mgh"
        )
        return read_mgh(file)[0]


@dataclass
class DataFiles:
    """Class containing file paths to subject data for retinotopy analysis."""

    subj: str  # subject name
    subject_id = "freesurfer"  # freesurfer subject id
    base_dir = Path(DIR_BASE)  # base directory

    @property
    def subjects_dir(self):
        """Directory to freesurfer subject."""
        return self.base_dir / self.subj / "anatomy"

    @property
    def dir_prf(self):
        """directory to prf data."""
        retinotopy = "retinotopy2" if self.subj == "p4" else "retinotopy"
        return self.base_dir / self.subj / retinotopy / "prf" / "map" / "sampled"

    @property
    def dir_benson(self):
        """directory to infferred retinotopy data."""
        retinotopy = "retinotopy2" if self.subj == "p4" else "retinotopy"
        return self.base_dir / self.subj / retinotopy / "prf" / "map" / "sampled_benson"

    def sampled_freesurfer(self, contrast, hemi, layer):
        """Sampled overlays in freesurfer space."""
        if not DIR_CACHE:
            raise ValueError("DIR_CACHE is not set in config!")

        vtx, _ = read_geometry(self.surf_freesurfer(hemi, "white"))
        ind = np.loadtxt(self.ind_dense(hemi), dtype=int)
        arr, _, _ = read_mgh(self.sampled_epi(contrast, hemi, layer))

        arr_cut = arr[ind < len(vtx)]
        ind_cut = ind[ind < len(vtx)]

        arr_freesurfer = np.zeros(len(vtx))
        arr_freesurfer[ind_cut] = arr_cut

        dir_out = Path(DIR_CACHE) / self.subj / self.subject_id / "surf"
        dir_out.mkdir(parents=True, exist_ok=True)

        file_out = dir_out / f"{hemi}.{contrast}_layer_{layer}.mgh"
        write_mgh(file_out, arr_freesurfer)
        return file_out

    def surf_dense(self, hemi, name):
        """white surface in upsampled (dense) freesurfer space."""
        return self.subjects_dir / "dense" / f"{hemi}.{name}"

    def surf_freesurfer(self, hemi, name):
        """white surface in freesurfer space."""
        return self.subjects_dir / self.subject_id / "surf" / f"{hemi}.{name}"

    def sampled_epi(self, contrast, hemi, layer):
        """sampled overlays of retinotopy data in epi space."""
        if contrast == "ecc":
            return self.dir_prf / "ecc" / f"{hemi}.ecc_layer_{layer}.mgh"
        elif contrast == "pol":
            return self.dir_prf / "pol" / f"{hemi}.pol_layer_{layer}.mgh"
        elif contrast == "rsq":
            return self.dir_prf / "high_rsq" / f"{hemi}.high_rsq_layer_{layer}.mgh"
        elif contrast == "prf_size":
            return self.dir_prf / "prf_size" / f"{hemi}.prf_size_layer_{layer}.mgh"
        else:
            return ValueError("Unknown contrast!")

    def sampled_epi_benson(self, contrast, hemi):
        """sampled inferred retinotopy in epi space."""
        if contrast == "ecc":
            return self.dir_benson / f"{hemi}.inferred_eccen.mgh"
        elif contrast == "pol":
            return self.dir_benson / f"{hemi}.inferred_angle.mgh"
        elif contrast == "area":
            return self.dir_benson / f"{hemi}.inferred_varea.mgh"
        elif contrast == "prf_size":
            return self.dir_benson / f"{hemi}.inferred_sigma.mgh"
        else:
            return ValueError("Unknown contrast!")

    def ind_dense(self, hemi):
        """index array for conversion from layer in epi space to dense."""
        return self.subjects_dir / "dense_refined" / f"{hemi}.ind2dense.txt"

    @property
    def subj(self):
        """Subject name."""
        return self._subj

    @subj.setter
    def subj(self, sub):
        if sub not in ["p1", "p2", "p3", "p4", "p5"]:
            raise ValueError("Unknown subj!")

        self._subj = sub
