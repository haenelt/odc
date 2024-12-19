# -*- coding: utf-8 -*-
"""Run decoding analysis with feature selected from time series from one layer. This is 
done for Shahin."""

import os
import functools
from pathlib import Path
import numpy as np

from fmri_decoder.data import DataConfig, ModelConfig, SurfaceData, TimeseriesData
from fmri_decoder.model import ExternalFeatureMVPA
from fmri_decoder.preprocessing import TimeseriesPreproc, TimeseriesSampling

from src.data import Data
from src.config import N_LAYER, DIR_DATA

__all__ = ["RunMVPA"]


DIR_CACHE = os.path.join(DIR_DATA, "cache_layer")


class RunMVPA:
    """Decoding analysis with shared features across cortical depth."""

    # Layer for feature sampling
    LAYER_FEATURE = 0

    def __init__(self, subj, seq, day, area, dir_out, verbose):
        self.subj = subj
        self.seq = seq
        self.day = day
        self.area = area  # v1, v2, v3, v2a, v2b, v3a or v3b
        self.verbose = verbose  # save samples to disk

        # make output directory
        self.dir_out = None
        self.dir_sample = None

        if dir_out is not None:
            self.dir_out = Path(dir_out)
            self.dir_out.mkdir(parents=True, exist_ok=True)
        if verbose is True and dir_out:
            self.dir_sample = self.dir_out / "sample"

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
    def n_surf(self):
        return len(self.surf_data.file_layer["lh"])

    @property
    @functools.lru_cache()
    def sess(self):
        return Data(self.subj, self.seq, self.day, self.area).sess

    @property
    def preprocessing(self):
        # get preprocessed volumetric time series and task events
        _file_vol = Path(DIR_CACHE) / f"data_vol_{self.subj}_{self.sess}.npy"
        _file_events = Path(DIR_CACHE) / f"events_{self.subj}_{self.sess}.npy"

        if _file_vol.is_file() and _file_events.is_file():
            _data_vol = np.load(_file_vol, allow_pickle=True)
            _events = np.load(_file_events, allow_pickle=True)
            return _data_vol, _events
        else:
            preproc = TimeseriesPreproc.from_dict(self.config)
            # detrend time series
            _ = preproc.detrend_timeseries(self.config_data.tr, self.config_data.cutoff_sec)
            # crop time series
            _data_vol, _events = preproc.crop_data(self.config_data.n_skip)

            # save dictionary to disk
            _file_vol.parent.mkdir(parents=True, exist_ok=True)
            np.save(_file_vol, _data_vol)
            np.save(_file_events, _events)

            return _data_vol, _events

    def data_feature_sampled(self, data_vol):
        # get features from time series averaged across cortical depth
        _file_data = Path(DIR_CACHE) / f"data_feature_sampled_{self.subj}_{self.sess}.npy"

        if _file_data.is_file():
            return np.load(_file_data, allow_pickle=True).item()
        else:
            _data = {}
            for hemi in ["lh", "rh"]:
                vtx, fac = self.surf_data.load_layer(hemi, self.LAYER_FEATURE)
                sampler = TimeseriesSampling(vtx, fac, data_vol)
                # sample time series
                file_deformation = self.config_data.file_deformation
                file_reference = self.time_data.file_series[0]
                _data[hemi] = sampler.sample_timeseries(
                    file_deformation, file_reference
                )

            for hemi in ["lh", "rh"]:
                label = self.surf_data.load_label_intersection(hemi)
                _data[hemi] = [
                    _data[hemi][x][label, :]
                    for x in range(len(_data[hemi]))
                ]

            # save dictionary to disk
            _file_data.parent.mkdir(parents=True, exist_ok=True)
            np.save(_file_data, _data)

            return _data
    
    def data_sampling(self, data_vol, layer):
        # sample time series data to surface
        _file_data = Path(DIR_CACHE) / f"data_sampled_{self.subj}_{self.sess}_layer{layer}.npy"

        if _file_data.is_file():
            return np.load(_file_data, allow_pickle=True).item()
        else:
            _data_sampled = {}
            for hemi in ["lh", "rh"]:
                vtx, fac = self.surf_data.load_layer(hemi, layer)
                sampler = TimeseriesSampling(vtx, fac, data_vol)
                # sample time series
                file_deformation = self.config_data.file_deformation
                file_reference = self.time_data.file_series[0]
                _data_sampled[hemi] = sampler.sample_timeseries(
                    file_deformation, file_reference
                )

            for hemi in ["lh", "rh"]:
                label = self.surf_data.load_label_intersection(hemi)
                _data_sampled[hemi] = [
                    _data_sampled[hemi][x][label, :]
                    for x in range(len(_data_sampled[hemi]))
                ]

            # save dictionary to disk
            _file_data.parent.mkdir(parents=True, exist_ok=True)
            np.save(_file_data, _data_sampled)

            return _data_sampled

    def decoding(self, save=False):
        data_vol, events = self.preprocessing

        # iterate over surfaces (layers)
        score = np.zeros(N_LAYER)
        for i in range(self.n_surf):
            data_sampled = self.data_sampling(data_vol, i)

            mvpa = ExternalFeatureMVPA.from_data(
                data_sampled,
                events,
                nmax=self.config_model.nmax,
                remove_nan=True,
                data_feature=self.data_feature_sampled(data_vol),
            )

            if self.verbose is True and self.dir_out is not None:
                self.dir_sample.mkdir(parents=True, exist_ok=True)
                mvpa.save_dataframe(self.dir_sample / f"sample_data_{i}.parquet")

            # model preparation and fitting
            # scaling
            if self.config_model.feature_scaling:
                mvpa.scale_features(self.config_model.feature_scaling)
            if self.config_model.sample_scaling:
                mvpa.scale_samples(self.config_model.sample_scaling)
            res = mvpa.evaluate

            # get scores
            score[i] += np.mean(res.accuracy)

            # save output
            if self.dir_out is not None:
                self.save()

        return score

    def save(self):
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
        "--sess",
        dest="sess",
        type=str,
        help="Session name (GE_EPI, SE_EPI, VASO, VASO_uncorrected).",
    )
    parser.add_argument("--day", dest="day", type=int, help="Session day (0 or 1).")
    parser.add_argument(
        "--area",
        dest="area",
        default="v1",
        type=str,
        help="Cortical area from which features are selected.",
    )
    parser.add_argument(
        "--ref", 
        dest="ref", 
        default=5,
        type=int, 
        help="Reference layer for feature selection.", 
        )
    parser.add_argument(
        "--save_samples",
        dest="save_samples",
        action="store_true",
        help="Save data samples to disk (default: %(default)s).",
        default=False,
    )
    args = parser.parse_args()

    # check arguments
    print("Running...")
    print(f"SUBJ: {args.subj}")
    print(f"SESSION: {args.sess}")
    print(f"DAY: {args.day}")
    print(f"AREA: {args.area}")
    print(f"REF: {args.ref}")
    print(f"Save samples: {args.save_samples}")

    dir_out = (
        Path(args.out)
        / "decoding"
        / args.subj
        / Data(args.subj, args.sess, args.day, args.area).sess
        / f"{args.area}_bandpass_none"
    )
    mvpa = RunMVPA(
        args.subj, args.sess, args.day, args.area, args.dir_out, args.save_samples
    )
    mvpa.LAYER_FEATURE = args.ref
    _ = mvpa.decoding()
