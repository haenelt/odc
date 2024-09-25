# -*- coding: utf-8 -*-
"""Run decoding analysis with feature selected from time series averaged across cortical
depth. This was one of the main reviewer comments."""

import functools
from pathlib import Path

from fmri_decoder.data import DataConfig, ModelConfig, SurfaceData, TimeseriesData
from fmri_decoder.model import ExternalFeatureMVPA
from fmri_decoder.preprocessing import TimeseriesPreproc, TimeseriesSampling

from src.data import Data

__all__ = ["RunMVPA"]


class RunMVPA:
    """Decoding analysis with shared features across cortical depth."""

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
    def preprocessing(self):
        # timeseries preprocessing
        preproc = TimeseriesPreproc.from_dict(self.config)
        # detrend time series
        _ = preproc.detrend_timeseries(self.config_data.tr, self.config_data.cutoff_sec)
        # crop time series
        data_vol, events = preproc.crop_data(self.config_data.n_skip)
        return data_vol, events

    def decoding(self, save=False):
        data_vol, events = self.preprocessing
        # get features from time series averaged across cortical depth
        n_surf = len(self.surf_data.file_layer["lh"])
        data_feature_sampled = {}
        for i in range(n_surf):
            for hemi in ["lh", "rh"]:
                vtx, fac = self.surf_data.load_layer(hemi, i)
                sampler = TimeseriesSampling(vtx, fac, data_vol)
                # sample time series
                file_deformation = self.config_data.file_deformation
                file_reference = self.time_data.file_series[0]
                if i == 0:
                    data_feature_sampled[hemi] = sampler.sample_timeseries(
                        file_deformation, file_reference
                    )
                else:
                    _tmp = sampler.sample_timeseries(file_deformation, file_reference)
                    data_feature_sampled[hemi] = [
                        a + b for a, b in zip(data_feature_sampled[hemi], _tmp)
                    ]

        for hemi in ["lh", "rh"]:
            data_feature_sampled[hemi] = [
                data_feature_sampled[hemi][x] / n_surf
                for x in range(len(data_feature_sampled[hemi]))
            ]

        for hemi in ["lh", "rh"]:
            label = self.surf_data.load_label_intersection(hemi)
            data_feature_sampled[hemi] = [
                data_feature_sampled[hemi][x][label, :]
                for x in range(len(data_feature_sampled[hemi]))
            ]

        # iterate over surfaces (layers)
        n_surf = len(self.surf_data.file_layer["lh"])
        score = np.zeros(N_LAYER)
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
                    data_sampled[hemi][x][label, :]
                    for x in range(len(data_sampled[hemi]))
                ]

            mvpa = ExternalFeatureMVPA.from_data(
                data_sampled,
                events,
                nmax=self.config_model.nmax,
                remove_nan=True,
                data_feature=data_feature_sampled,
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
    _ = mvpa.decoding()
