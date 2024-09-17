# -*- coding: utf-8 -*-
"""Filepaths to data."""

import re
from dataclasses import dataclass
from pathlib import Path

from fmri_tools.io.surf import read_mgh

from .config import DIR_BASE, N_LAYER, N_RUN, SESSION

__all__ = ["Data"]


@dataclass
class Data:
    """File paths to subject data."""

    subj: str
    sequence: str
    day: int
    area: str

    @property
    def sess(self):
        """Session name."""
        if "_uncorrected" in self.sequence:
            return f"{self.sequence[: 4]}{SESSION[self.subj][self.sequence][self.day]}_uncorrected"
        return f"{self.sequence}{SESSION[self.subj][self.sequence][self.day]}"

    @property
    def surfaces(self):
        """File names of surface geometries."""
        file_ = {}
        for hemi in ["lh", "rh"]:
            file_[hemi] = [
                str(
                    Path(DIR_BASE)
                    / self.subj
                    / "anatomy"
                    / "layer"
                    / f"{hemi}.layer_{i}"
                )
                for i in range(N_LAYER)
            ]
        return file_

    @property
    def labels(self):
        """File names of labels."""
        file_ = {}
        for hemi in ["lh", "rh"]:
            file_[hemi] = [
                str(
                    Path(DIR_BASE)
                    / self.subj
                    / "anatomy"
                    / "label_benson"
                    / f"{hemi}.{self.area}.label"
                ),
                str(
                    Path(DIR_BASE)
                    / self.subj
                    / "anatomy"
                    / "label"
                    / f"{hemi}.fov.label"
                ),
            ]
        return file_

    @property
    def timeseries(self):
        """File names of fmri time series."""
        if "VASO" in self.sess and "uncorrected" in self.sess:
            sess_ = re.sub("_uncorrected", "", self.sess)
            file_ = [
                str(
                    Path(DIR_BASE)
                    / self.subj
                    / "odc"
                    / sess_
                    / f"Run_{i + 1}"
                    / "ubold_upsampled.nii"
                )
                for i in range(N_RUN)
            ]
        elif "VASO" in self.sess:
            file_ = [
                str(
                    Path(DIR_BASE)
                    / self.subj
                    / "odc"
                    / self.sess
                    / f"Run_{i + 1}"
                    / "uvaso_upsampled_corrected.nii"
                )
                for i in range(N_RUN)
            ]
        else:
            file_ = [
                str(
                    Path(DIR_BASE)
                    / self.subj
                    / "odc"
                    / self.sess
                    / f"Run_{i + 1}"
                    / "udata.nii"
                )
                for i in range(N_RUN)
            ]
        return file_

    @property
    def events(self):
        """File names of condition files."""
        sess_ = (
            re.sub("_uncorrected", "", self.sess)
            if "_uncorrected" in self.sess
            else self.sess
        )
        file_ = [
            str(
                Path(DIR_BASE)
                / self.subj
                / "odc"
                / sess_
                / f"Run_{i+1}"
                / "logfiles"
                / f"{self.subj}_{sess_}_Run{i + 1}_odc_Cond.mat"
            )
            for i in range(N_RUN)
        ]
        return file_

    @property
    def deformation(self):
        """File name of coordinate mapping."""
        sess_ = (
            re.sub("_uncorrected", "", self.sess)
            if "_uncorrected" in self.sess
            else self.sess
        )
        file_ = str(
            Path(DIR_BASE)
            / self.subj
            / "deformation"
            / "odc"
            / sess_
            / "source2target.nii.gz"
        )
        return file_

    def get_sample_data(self, layer, version):
        """Load sample data from MVPA analysis."""
        _dirname = "bandpass_none" if version == "v1.0" else f"{self.area}_bandpass_none"
        file_ = (
            Path(DIR_BASE)
            / "paper"
            / version
            / "decoding"
            / self.subj
            / self.sess
            / _dirname
            / "sample"
            / f"sample_data_{layer}.parquet"
        )
        return file_

    def get_contrast(self, hemi, layer):
        """Load z-contrast left > right."""
        file_ = (
            Path(DIR_BASE)
            / self.subj
            / "odc"
            / "results"
            / "Z"
            / "sampled"
            / f"Z_all_left_right_{self.sess}"
            / f"{hemi}.Z_all_left_right_{self.sess}_layer_{layer}.mgh"
        )
        return read_mgh(file_)[0]

    def get_tsnr(self, hemi, layer):
        """Load tsnr map."""
        file_ = (
            Path(DIR_BASE)
            / self.subj
            / "odc"
            / "results"
            / "tsnr"
            / "sampled"
            / f"tsnr_all_{self.sess}"
            / f"{hemi}.tsnr_all_{self.sess}_layer_{layer}.mgh"
        )
        return read_mgh(file_)[0]

    def get_psc(self, hemi, layer):
        """Load percent signal change averaged over left > rest and right > rest."""
        file_left = (
            Path(DIR_BASE)
            / self.subj
            / "odc"
            / "results"
            / "raw"
            / "sampled"
            / f"psc_all_left_rest_{self.sess}"
            / f"{hemi}.psc_all_left_rest_{self.sess}_layer_{layer}.mgh"
        )
        file_right = (
            Path(DIR_BASE)
            / self.subj
            / "odc"
            / "results"
            / "raw"
            / "sampled"
            / f"psc_all_right_rest_{self.sess}"
            / f"{hemi}.psc_all_right_rest_{self.sess}_layer_{layer}.mgh"
        )
        return (read_mgh(file_left)[0] + read_mgh(file_right)[0]) / 2

    def get_cnr(self, hemi, layer):
        """Load contrast-to-noise ratio averaged over left > rest and right > rest."""
        file_left = (
            Path(DIR_BASE)
            / self.subj
            / "odc"
            / "results"
            / "cnr"
            / "sampled"
            / f"cnr_all_left_rest_{self.sess}"
            / f"{hemi}.cnr_all_left_rest_{self.sess}_layer_{layer}.mgh"
        )
        file_right = (
            Path(DIR_BASE)
            / self.subj
            / "odc"
            / "results"
            / "cnr"
            / "sampled"
            / f"cnr_all_right_rest_{self.sess}"
            / f"{hemi}.cnr_all_right_rest_{self.sess}_layer_{layer}.mgh"
        )
        return (read_mgh(file_left)[0] + read_mgh(file_right)[0]) / 2

    def get_rest(self, hemi, layer):
        """Load resting-state data."""
        name_rest = "resting_state2" if self.subj == "p4" else "resting_state"
        file_ = (
            Path(DIR_BASE)
            / self.subj
            / name_rest
            / "ssw"
            / "sampled"
            / "nssw"
            / f"{hemi}.nssw_layer_{layer}.mgh"
        )
        return read_mgh(file_)[0]
