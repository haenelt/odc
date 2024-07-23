# -*- coding: utf-8 -*-
"""Some reprocessing for reviewer."""

import functools
import re
from pathlib import Path

import numpy as np
import pandas as pd
from fmri_decoder.data import DataConfig, ModelConfig, SurfaceData, TimeseriesData
from fmri_decoder.model import MVPA
from fmri_decoder.preprocessing import TimeseriesPreproc, TimeseriesSampling
from sklearn.feature_selection import f_classif

from .config import DIR_BASE, N_LAYER, N_RUN, SESSION

__all__ = ["Data", "FeatureSelect", "RunMVPA"]


class Data:
    """File paths to (reprocessed) data for review. More specific, label files now point
    to the labels generated with the Benson method and sample data point to the version
    v2.0."""

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


class FeatureSelect:
    """Select features based on an already sampled data set."""

    def __init__(self, subj, sess, day, area):
        self.subj = subj
        self.sess = sess
        self.day = day
        self.area = area
        self.data = Data(self.subj, self.sess, self.day, self.area)
        self.label, self.hemi = self.get_label
        self.label_sorted, self.hemi_sorted = zip(
            *[self.sort_features(i) for i in range(N_LAYER)]
        )

    @property
    @functools.lru_cache()
    def get_label(self):
        """Get label and hemisphere."""
        surf_data = SurfaceData(self.data.surfaces, None, self.data.labels)

        label_left = surf_data.load_label_intersection("lh")
        label_right = surf_data.load_label_intersection("rh")

        hemi = np.zeros(len(label_left) + len(label_right))
        hemi[len(label_left) :] = 1
        label = np.append(label_left, label_right)

        return label, hemi

    def sort_features(self, layer):
        """Sort label and hemi array based on features."""
        dtf = pd.read_parquet(self.data.get_sample_data(layer))

        # choose subset of features
        features = dtf.columns[2:]

        X = np.array(dtf.loc[:, features])
        y = np.array(dtf.loc[:, "label"])

        f_statistic = f_classif(X, y)[0]
        index = np.arange(len(features))
        index_sorted = np.array(
            [x for _, x in sorted(zip(f_statistic, index), reverse=True)]
        )

        label_sorted = self.label[index_sorted]
        hemi_sorted = self.hemi[index_sorted]

        return label_sorted, hemi_sorted


class RunMVPA:
    """Decoding analysis with shared features across cortical depth."""

    def __init__(self, dir_out, subj, seq, day, area, feature_layer=5):
        self.subj = subj
        self.seq = seq
        self.day = day
        self.area = area  # v1, v2, v3, v2a, v2b, v3a or v3b
        self.feature_layer = feature_layer  # layer for feature selection

        # make output directory
        self.dir_out = Path(dir_out)
        self.dir_out.mkdir(parents=True, exist_ok=True)

        # load data
        self.time_data = TimeseriesData.from_dict(self.config)
        self.surf_data = SurfaceData.from_dict(self.config)
        self.config_data = DataConfig.from_dict(self.config)
        self.config_model = ModelConfig.from_dict(self.config)

    @property
    @functools.lru_cache()
    def config(self):
        data = Data(self.subj, self.seq, self.day, self.area)
        _config = {}
        _config["TR"] = 3
        _config["n_skip"] = 2
        _config["cutoff_sec"] = 270
        _config["filter_size"] = None
        _config["nmax"] = 200
        _config["radius"] = None
        _config["feature_scaling"] = "standard"
        _config["sample_scaling"] = None
        _config["file_series"] = data.timeseries
        _config["file_events"] = data.events
        _config["file_layer"] = data.surfaces
        _config["file_deformation"] = data.deformation
        _config["file_localizer"] = None
        _config["file_label"] = data.labels
        _config["randomize_labels"] = False
        return _config

    @property
    @functools.lru_cache()
    def feature_selection(self):
        features = FeatureSelect(self.subj, self.seq, self.day, self.area)
        _features_selected = {
            "hemi": features.hemi_sorted[self.feature_layer][: self.config_model.nmax],
            "label": features.label_sorted[self.feature_layer][
                : self.config_model.nmax
            ],
        }
        return _features_selected

    @property
    @functools.lru_cache()
    def preprocessing(self):
        # timeseries preprocessing
        preproc = TimeseriesPreproc.from_dict(self.config)
        # detrend time series
        _ = preproc.detrend_timeseries(self.config_data.tr, self.config_data.cutoff_sec)
        # crop time series
        data_vol, events = preproc.crop_data(self.config_data.n_skip)
        return data_vol, events

    def decoding(self):
        data_vol, events = self.preprocessing
        # iterate over surfaces (layers)
        n_surf = len(self.surf_data.file_layer["lh"])
        for i in range(n_surf):
            data_sampled = {}
            for hemi in ["lh", "rh"]:
                vtx, fac = self.surf_data.load_layer(hemi, i)
                sampler = TimeseriesSampling(vtx, fac, data_vol)
                # sample time series
                file_deformation = self.config_data.file_deformation
                file_reference = self.time_data.file_series[0]
                data_sampled[hemi] = sampler.sample_timeseries(
                    file_deformation, file_reference
                )

            for hemi in ["lh", "rh"]:
                label = self.surf_data.load_label_intersection(hemi)
                data_sampled[hemi] = [
                    data_sampled[hemi][x][label, :] for x in range(len(data_sampled[hemi]))
                ]
            
            mvpa = MVPA.from_data(
                data_sampled, events, nmax=self.config_model.nmax, remove_nan=True
            )

            # model preparation and fitting
            # scaling
            if self.config_model.feature_scaling:
                mvpa.scale_features(self.config_model.feature_scaling)
            if self.config_model.sample_scaling:
                mvpa.scale_samples(self.config_model.sample_scaling)
            _ = mvpa.evaluate

            # save results
            mvpa.save_results(self.dir_out / "accuracy.csv", "accuracy")
            mvpa.save_results(self.dir_out / "sensitivity.csv", "sensitivity")
            mvpa.save_results(self.dir_out / "specificity.csv", "specificity")
            mvpa.save_results(self.dir_out / "f1.csv", "f1")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(description="Run MVPA.")
    parser.add_argument("--out", dest="out", type=str, help="Output base directory.")
    parser.add_argument("--subj", dest="subj", type=str, help="Subject name.")
    parser.add_argument(
        "--area",
        dest="area",
        default="v1",
        type=str,
        help="Cortical area from which features are selected.",
    )
    parser.add_argument(
        "--flayer",
        dest="feature_layer",
        default=5,
        type=int,
        help="Layer from which features are selected.",
    )
    args = parser.parse_args()

    # check arguments
    print(f"AREA: {args.area}")
    print(f"FEATURE LAYER: {args.feature_layer}")

    for seq in ["GE_EPI", "SE_EPI", "VASO"]:
        for day in range(2):
            print(f"Running: subj -> {args.subj}, sequence -> {seq}, day -> {day}")
            dir_out = (
                Path(args.out)
                / args.subj
                / Data(args.subj, seq, day, args.area).sess
                / f"{args.area}_bandpass_none"
            )
            mvpa = RunMVPA(dir_out, args.subj, seq, day, args.area, args.feature_layer)
            mvpa.decoding()
