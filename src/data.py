# -*- coding: utf-8 -*-
"""Filepaths to data."""

import re
from dataclasses import dataclass
from pathlib import Path

from fmri_tools.io.surf import read_mgh

from .config import DIR_BASE, N_LAYER, N_RUN, SESSION

__all__ = ["Data", "DataNew"]


@dataclass
class Data:
    """Class for loading of sampled data."""

    subj: str
    session: str
    DIR_BASE = Path("/data/pt_01880/Experiment1_ODC/")

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
        """Load label files. Changed to load benson labels files."""
        file_ = {}
        file_["lh"] = [
            str(self.DIR_BASE / self.subj / "anatomy" / "label" / "lh.fov.label"),
            str(self.DIR_BASE / self.subj / "anatomy" / "label_benson" / "lh.v1.label"),
        ]
        file_["rh"] = [
            str(self.DIR_BASE / self.subj / "anatomy" / "label" / "rh.fov.label"),
            str(self.DIR_BASE / self.subj / "anatomy" / "label_benson" / "rh.v1.label"),
        ]
        return file_

    def get_sample_data(self, layer):
        """Load sample data from MVPA analysis."""
        file = (
            self.DIR_BASE
            / "paper"
            / "decoding"
            / self.subj
            / self.session
            / "bandpass_none"
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


class DataNew:
    """File paths to (reprocessed) data for review. More specific, label files now point
    to the labels generated with the Benson method and sample data point to the version
    v3.0."""

    def __init__(self, subj, sequence, day, area):
        self.subj = subj
        self.sess = f"{sequence}{SESSION[self.subj][sequence][day]}"
        self.day = day
        self.area = area

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

    def get_sample_data(self, layer):
        """Load sample data from MVPA analysis."""
        file = (
            Path(DIR_BASE)
            / "paper"
            / "v3.0"
            / self.subj
            / self.sess
            / f"{self.area}_bandpass_none"
            / "sample"
            / f"sample_data_{layer}.parquet"
        )
        return file

    def get_sample_data_v2p0(self, layer):
        """Load sample data from MVPA analyssi (v2.0)."""
        file = (
            Path(DIR_BASE) 
            / "paper" 
            / "v2.0" 
            / "decoding" 
            / self.subj 
            / self.sess 
            / f"{self.area}_bandpass_none" 
            / "sample" 
            / f"sample_data_{layer}.parquet"
        )
        return file

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