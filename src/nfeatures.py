# -*- coding: utf-8 -*-
"""Run number of features: python run_features.py --subj <subject name> 
--seq <sequence name> --day <day>"""

import os
from pathlib import Path
import gc

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.mvpa import RunMVPA
from src.config import DIR_BASE, N_LAYER
from src.data import Data

__all__ = ["RunNFeatures"]


# Constants
NUM_CORES = 128
os.environ["OMP_NUM_THREADS"] = 1  # limit numpy to single threads


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

        dir_out = (
            Path(DIR_BASE)
            / "paper"
            / self.version
            / "n_features"
            / self.subj
            / self.session
        )
        dir_out.mkdir(parents=True, exist_ok=True)

    def _compute(self, i):
        mvpa = RunMVPA(self.subj, self.sess, self.day, self.area, None, False)
        mvpa.condig["nmax"] = i
        score = mvpa.decoding()
        # garbage collection
        gc.collect()
        return score

    def run(self):
        _res = Parallel(n_jobs=NUM_CORES, backend="loky")(
            delayed(self._compute)(_i) for _i in tqdm(range(1, self.nmax + 1))
        )
        return _res

    def save(self):
        # save as csv
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
