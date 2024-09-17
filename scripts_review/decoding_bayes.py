# -*- coding: utf-8 -*-
"""
Bayesian model comparison

In this notebook, I fit the cortical profiles of decoding accuracies with a linear model
and a sine model using Bayesian inference. This is done for Bayesian model comparisons
by estimating the WAIC (Watanabe-Akaike information criterion) parameter. Decoding
profiles increases toward the cortical surface similar to percent signal changes.
Therefore, a linear model would be appropriate to describe the data. However, if the
peak in deeper layers is meaningful, i.e. can be related to the thalamocortical input
from LGN, a sinusoidal model might capture the peak in deeper layers. A similar model
comparison was performend in De Hollander et al., (Neuroimage, 2021) to compare linear
and quadratic models for decoding profiles.

Due to the few and noisy data, prior distributions of frequency and phase lag for the
sine model were assumed to be rather tight and therefore have the risk to overfit the
data. Therefore, interpretation should be cautious. The frequency and phase of the sine
curve were adjusted to have a peak in deeper layers as expected from neurophysiology.

Results for reviewers (not used in the response letter and just listed for completeness)
- layer 0 -> white, 10 -> pial
- v1-v3 distance (layer: 0; new)
  - mean: 7.79 mm
  - min: 0.90 mm
  - std: 2.61 mm
- v1-v3 distance (layer: 10; new)
  - mean: 9.47 mm
  - min: 1.27 mm
  - std: 3.07 mm
- filter sizes (mean period)
  - 0.025: 1.09
  - 0.05: 1.84
  - 0.1: 2.62
  - 0.5: 5.71
- Bayesian model comparison (WAIC; old)
  - GE_EPI: 319.83 (linear), 332.27 (sine)
  - SE_EPI: 356.82 (linear), 365.66 (sine)
  - VASO: 297.61 (linear), 298.76 (sine)
- Bayesian model comparison (WAIC; new)
  - GE_EPI: 331.58 (linear), 343.09 (sine)
  - SE_EPI: 357.71 (linear), 365.95 (sine)
  - VASO: 285.81 (linear), 290.69 (sine)
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.bayes import LinearModel, SineModel


def _main(sess, area, version):
    print("Running linear model...")
    lin_mod = LinearModel.from_data(sess, area, version)
    lin_mod.sample()
    lin_mod.plot_traces()
    lin_mod.plot_fit()
    lin_mod.evaluate_model("waic")

    print("Running sine model...")
    cos_mod = SineModel.from_data(sess, area, version)
    cos_mod.sample()
    cos_mod.plot_traces()
    cos_mod.plot_fit()
    cos_mod.evaluate_model("waic")


if __name__ == "__main__":
    import argparse

    # add argument
    parser = argparse.ArgumentParser(
        description="Bayesian model comparison.",
    )
    parser.add_argument(
        "--sess",
        dest="sess",
        type=str,
        help="Session name (GE_EPI, SE_EPI, VASO, VASO_uncorrected).",
    )
    parser.add_argument(
        "--area",
        dest="area",
        default="v1",
        type=str,
        help="Cortical area from which features are selected (e.g. v1).",
    )
    parser.add_argument(
        "--version",
        dest="version",
        default="v3.0",
        type=str,
        help="Analysis version.",
    )
    args = parser.parse_args()

    # run
    _main(args.sess, args.area, args.version)
