# -*- coding: utf-8 -*-
"""Bayesian modelling of subject retinotopy."""

from pathlib import Path

import neuropythy as ny
import numpy as np
from fmri_tools.io.surf import read_mgh, write_mgh
from neuropythy.commands.register_retinotopy import main as register_retinotopy
from nibabel.freesurfer.io import read_geometry
from scipy.interpolate import griddata

from .config import DIR_CACHE
from .data import DataFiles

__all__ = ["BayesianRetinotopy"]


class BayesianRetinotopy:
    """Bayesian analysis of subject retinotopy.

    This class applies the `register_retinotopy` command from the neuropythy library to
    my subject data, which registers a retinotopy template to subject retinotopy data
    using Bayesian inference. Please have a look at the References for more information.

    The neuropythy library provides an api which lets you run the command from the
    terminal:

    python -m neuropythy \
        register_retinotopy <subject_id> \
        --verbose \
        --lh-angle=<lh-polar-angle-data> \
        --lh-eccen=<lh-eccentricity-data> \
        --lh-radius=<lh-prf-size-data> \
        --lh-weight=<lh-explained-variance-data> \
        --rh-angle=<rh-polar-angle-data> \
        --rh-eccen=<rh-eccentricity-data> \
        --rh-radius=<rh-prf-size-data> \
        --rh-weight=<rh-explained-variance-data>

    To run this code, you need to have sampled the data retinotopy onto a surface mesh
    and have a freesurfer segmentation folder for this subject. Since, I transformed the
    surfaces after running the freesurfer segmentation, I needed to adapt the
    computation to the data of the current project. This is the main purpose of this
    class. It wraps the `register_retinotopy` command and transforms the data to the
    space that matches all other data of this project.

    As a last note, the `register_retinotopy` command expects retinotopy data in a
    specific data range:

    - lh-angle: subject's LH polar angle, 0 -> +180 degrees refers to UVM -> RHM -> LVM
    - rh-angle: subject's RH polar angle, 0 -> -180 degrees refers to UVM -> LHM -> LVM
    - lh-eccen: subject's LH eccentricity, in degrees from the fovea
    - rh-eccen: subject's RH eccentricity, in degrees from the fovea
    - lh-variance-explained: the variance explained of each vertex's pRF solution for
      the LH; 0-1 values
    - rh-variance-explained: the variance explained of each vertex's pRF solution for
      the RH; 0-1 values
    - lh-radius: prf size for the LH in degrees of the visual field
    - rh-radius: prf size for the RH in degrees of the visual field

    References
    ----------
    .. [1] N.C. Benson, J., Winawer. Bayesian analysis of retinotopic maps, eLife,
           vol. 7, e40224, 2018.
    .. [2] https://nben.net/Retinotopy-Tutorial/#bayesian-maps
    .. [3] https://github.com/noahbenson/neuropythy-tutorials
    .. [4] https://github.com/noahbenson/neuropythy/wiki

    Attributes
    ----------
    subj : str
        Subject name.
    ecc_min : float
        Minimum eccentricity value in deg.
    ecc_max : float
        Maximum eccentricity value in deg.
    rsq_min : float
        Explained variance threshold.
    layer : int
        Layer number.
    """

    # Output directory for all generated data that are not in the desired data format.
    dir_cache = DIR_CACHE

    def __init__(self, subj, ecc_min=0, ecc_max=6, rsq_min=0.1, layer=5):
        self.subj = subj
        self.ecc_min = ecc_min
        self.ecc_max = ecc_max
        self.rsq_min = rsq_min
        self.layer = layer

        # get all necessary file paths for subj
        self.files = DataFiles(self.subj)

        # set neuropythy configuration
        ny.config["freesurfer_subject_paths"] = str(self.files.subjects_dir)
        data_cache_root = Path(BayesianRetinotopy.dir_cache)
        data_cache_root.mkdir(parents=True, exist_ok=True)
        ny.config["data_cache_root"] = data_cache_root

        # define output folders for neuropythy output
        self.dir_out = data_cache_root / self.files.subj / self.files.subject_id
        self.dir_surf = self.dir_out / "surf"
        self.dir_vol = self.dir_out / "vol"

    @property
    def _command(self):
        """Arguments used by `register_retinotopy`. Default values are used for --scale,
        --max-steps, --max-step-size and --prior. --scale sets the strength of the
        functional forces relative to anatomical forces. --max-steps sets the maximum
        number of steps to run the registration. --max-step-size sets the maximum
        step-size for any single vertx. --prior sets the used prior template."""
        lh_angle = (
            f"--lh-angle={self.files.sampled_freesurfer('pol', 'lh', self.layer)}"
        )
        lh_eccen = (
            f"--lh-eccen={self.files.sampled_freesurfer('ecc', 'lh', self.layer)}"
        )
        lh_radius = (
            f"--lh-radius={self.files.sampled_freesurfer('prf_size', 'lh', self.layer)}"
        )
        lh_weight = (
            f"--lh-weight={self.files.sampled_freesurfer('rsq', 'lh', self.layer)}"
        )
        rh_angle = (
            f"--rh-angle={self.files.sampled_freesurfer('pol', 'rh', self.layer)}"
        )
        rh_eccen = (
            f"--rh-eccen={self.files.sampled_freesurfer('ecc', 'rh', self.layer)}"
        )
        rh_radius = (
            f"--rh-radius={self.files.sampled_freesurfer('prf_size', 'rh', self.layer)}"
        )
        rh_weight = (
            f"--rh-weight={self.files.sampled_freesurfer('rsq', 'rh', self.layer)}"
        )
        weight_min = f"--weight-min={self.rsq_min}"
        max_output_eccen = f"--max-output-eccen={self.ecc_max}"  # in degrees
        max_input_eccen = f"--max-input-eccen={self.ecc_max}"  # in degrees
        min_input_eccen = f"--min-input-eccen={self.ecc_min}"  # in degrees
        surf_outdir = f"--surf-outdir={self.dir_surf}"
        vol_outdir = f"--vol-outdir={self.dir_vol}"

        return (
            self.files.subject_id,
            "--verbose",
            lh_angle,
            lh_eccen,
            lh_radius,
            lh_weight,
            rh_angle,
            rh_eccen,
            rh_radius,
            rh_weight,
            weight_min,
            "--scale=20",
            "--max-steps=2000",
            "--max-step-size=0.05",
            "--prior=benson17",
            max_output_eccen,
            max_input_eccen,
            min_input_eccen,
            "--vol-format=mgz",
            "--surf-format=mgh",
            vol_outdir,
            surf_outdir,
        )

    def run_registration(self, dir_out):
        """Executation of the `register_retinotopy` command and transformation of
        output data to the desired target space.

        Parameters
        ----------
        dir_out : str or pathlib.Path class instance
            Output directory of inferred retinotopy in target space.

        Returns
        -------
        None.
        """
        # create output folders for the output generated by the register_retinotopy
        # command
        self._create_output_folders()

        # run
        register_retinotopy(self._command)

        # upsample results to dense space
        dir_cache_in = (
            Path(BayesianRetinotopy.dir_cache)
            / self.files.subj
            / self.files.subject_id
            / "surf"
        )
        dir_cache_out = (
            Path(BayesianRetinotopy.dir_cache)
            / self.subj
            / self.files.subject_id
            / "surf_dense_benson"
        )
        for fid in dir_cache_in.glob("*inferred*mgh"):
            self._surf_freesurfer_to_dense(fid, dir_cache_out)

        # transform upsampled results to epi space
        for fid in dir_cache_out.glob("*inferred*mgh"):
            self._surf_dense_to_epi(fid, dir_out)

    def _create_output_folders(self):
        """Make output folders for the `register_retinotopy` output.

        Returns
        -------
        None.
        """
        self.dir_surf.mkdir(parents=True, exist_ok=True)
        self.dir_vol.mkdir(parents=True, exist_ok=True)

    def _surf_freesurfer_to_dense(self, mgh_in, dir_out, method="nearest"):
        """Interpolate overlays to upsampled (dense) mesh.

        Parameters
        ----------
        mgh_in : str or pathlib.Path class instance
            File name of input overlay.
        dir_out : str or pathlib.Path class instance
            Output directory of saved upsampled overlay.
        method : str, optional
            Interpolation method (nearest, linear, cubic).

        Returns
        -------
        None.
        """
        _hemi = Path(mgh_in).name[:2]
        pts_sphere_dense, _ = read_geometry(self.files.surf_dense(_hemi, "sphere"))
        pts_sphere, _ = read_geometry(self.files.surf_freesurfer(_hemi, "sphere"))
        arr, _, _ = read_mgh(mgh_in)
        arr_dense = griddata(pts_sphere, arr, pts_sphere_dense, method)
        write_mgh(dir_out / Path(mgh_in).name, arr_dense)

    def _surf_dense_to_epi(self, mgh_in, dir_out):
        """Transform overlay to epi space.

        Parameters
        ----------
        mgh_in : str or pathlib.Path class instance
            File name of input overlay.
        dir_out : str or pathlib.Path class instance
            Output directory of saved transformed overlay.

        Returns
        -------
        None.
        """
        _hemi = Path(mgh_in).name[:2]
        ind = np.loadtxt(self.files.ind_dense(_hemi), dtype=int)
        arr, _, _ = read_mgh(mgh_in)
        write_mgh(dir_out / Path(mgh_in).name, arr[ind])


if __name__ == "__main__":
    # run from root directory with python -m psf_bold.retinotopy

    # parameters
    SUBJ = "p1"  # subject name
    LAYER = 5  # layer used for computation
    RSQ_MIN = 0.1  # treshold for data during retinotopy registration
    ECC_MIN = 0.0  # minimum eccentricity
    ECC_MAX = 6.0  # maximum eccentricity
    N_LAYER = 11  # number of layers for which iso-eccentricity distances are computed

    # file paths to subject data
    files = DataFiles(SUBJ)

    # register retinotopy
    bayes = BayesianRetinotopy(SUBJ, ECC_MIN, ECC_MAX, RSQ_MIN, LAYER)
    bayes.run_registration(files.dir_benson)
