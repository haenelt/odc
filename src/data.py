from dataclasses import dataclass
from pathlib import Path

from fmri_tools.io.surf import read_mgh

from .config import N_LAYER


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
