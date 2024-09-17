# -*- coding: utf-8 -*-
"""Run number of features: python run_features.py --subj <subject name> 
--seq <sequence name> --day <day>"""

from pathlib import Path

import numpy as np
from fmri_decoder.model import MVPA
from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import DIR_BASE, N_LAYER
from src.data import Data

__all__ = ["RunNFeatures"]


# Constants
NUM_CORES = 128


class RunNFeatures:
    """Compute accuracies for various numbers of features."""

    def __init__(self, subj, seq, day, nmax, version):
        self.subj = subj
        self.seq = seq
        self.day = day
        self.nmax = nmax
        self.version = version
        self.data = Data(subj, seq, day, "v1")
        self.session = self.data.sess

    def _compute(self, i):
        score = np.zeros(N_LAYER)
        for layer in range(N_LAYER):
            file_sample = self.data.get_sample_data(layer, self.version)
            mvpa = MVPA.from_file(file_sample, nmax=i)
            mvpa.scale_features("standard")
            res = mvpa.evaluate
            score[layer] += np.mean(res.accuracy)
        return score

    def run(self):
        _res = Parallel(n_jobs=NUM_CORES)(
            delayed(self._compute)(_i) for _i in tqdm(range(1, self.nmax + 1))
        )
        return _res

    def save(self):
        # save as csv
        dir_out = (
            Path(DIR_BASE)
            / "paper"
            / self.version
            / "n_features"
            / self.subj
            / self.session
        )
        dir_out.mkdir(parents=True, exist_ok=True)
        np.savetxt(dir_out / "accuracy.csv", self.run(), delimiter=",")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run MVPA for different numbers of features."
    )
    parser.add_argument("--subj", dest="subj", type=str, help="Subject name.")
    parser.add_argument(
        "--sess",
        dest="sess",
        type=str,
        help="Session name (GE_EPI, SE_EPI, VASO, VASO_uncorrected).",
    )
    parser.add_argument("--day", dest="day", type=int, help="Session day (0 or 1).")
    parser.add_argument(
        "--nmax", dest="nmax", default=200, type=int, help="Number of features.",
    )
    parser.add_argument(
        "--version", dest="version", default="v3.0", type=str, help="Analysis version.",
    )
    args = parser.parse_args()

    # check arguments
    print("Running...")
    print(f"SUBJ: {args.subj}")
    print(f"SESSION: {args.sess}")
    print(f"DAY: {args.day}")
    print(f"NMAX: {args.nmax}")
    print(f"VERSION: {args.version}")

    nfeature = RunNFeatures(args.subj, args.sess, args.day, args.nmax, args.version)
    nfeature.save()

    print("Done.")
