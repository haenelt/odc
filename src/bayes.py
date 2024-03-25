# -*- coding: utf-8 -*-
"""Class for bayesian model comparison."""

import os
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from palettable.matplotlib import Inferno_3 as ColMap
from pymc.model_graph import model_to_graphviz

from src.config import DIR_DATA, N_LAYER, SESSION, SUBJECTS

plt.style.use(os.path.join(os.path.dirname(__file__), "default.mplstyle"))

__all__ = ["LinearModel", "SineModel"]


class LinearModel:
    """Linear regression model using Bayesian inference.

    Parameters
    ----------
    x : (N,) np.ndarray
        Array of cortical layers.
    y : (N,) np.ndarray
        Array of corresponding decoding accuracies in %.
    group : (N,) np.ndarray
        Array of corresponding integers indicating from which subject the data was
        pooled. Note that in the current version, this parameter is neglected, since
        a pooled Bayesian model is applied.

    """

    def __init__(self, x, y, group):
        self.x = x
        self.y = y
        self.group = group
        self.model = pm.Model()
        self.trace = None  # model trace
        self.init = False  # True if pymc model was alreay initalized

    @property
    def mean_data(self):
        """Compute mean across subjects of input data.

        Returns
        -------
        (np.ndarray, np.ndarray)
            Tuple containing averaged x- and y-values.
        """
        x_data = np.zeros(N_LAYER)
        y_data = np.zeros(N_LAYER)
        for i in range(N_LAYER):
            x_data[i] = i
            y_data[i] = np.mean(self.y[np.where(self.x == i)[0]])
        return x_data, y_data

    def init_model(self):
        """Initialize pymc model. This method therefore contains the prior settings."""
        if self.init:
            raise ValueError("Model already initialized!")
        with self.model:
            b0 = pm.Normal("b0", mu=0, sigma=100)  # interception
            b1 = pm.Normal("b1", mu=0, sigma=100)  # slope

            # define Linear model
            yest = self._fun(self.x, (b0, b1), "pymc")

            # define Normal likelihood with HalfCauchy noise
            y_sigma = pm.HalfCauchy("y_sigma", beta=10)
            likelihood = pm.Normal(  # noqa: F841
                "likelihood", mu=yest, sigma=y_sigma, observed=self.y
            )

        self.init = True

    def sample(self):
        """Compute sample trace via MCMC sampling."""
        if not self.init:
            self.init_model()
        with self.model:
            self.trace = pm.sample(
                draws=10000,
                tune=500,
                chains=4,
                target_accept=0.95,
                progressbar=True,
                return_inferencedata=True,
                idata_kwargs={"log_likelihood": True},
            )

    def plot_prior(self):
        """Plot predicitive priors."""
        if not self.init:
            self.init_model()
        with self.model:
            prior_checks = pm.sample_prior_predictive(samples=500)
        b0 = prior_checks.prior.b0.to_numpy()[0]
        b1 = prior_checks.prior.b1.to_numpy()[0]
        _params = (b0, b1)
        self._plot_helper(_params)

    def plot_fit(self):
        """Plot resulting data fit."""
        if self.trace is None:
            raise ValueError("No sampling done!")
        b1 = np.mean(self.trace.posterior.b1.to_numpy(), axis=0)
        b0 = np.mean(self.trace.posterior.b0.to_numpy(), axis=0)
        _params = (b0, b1)
        self._plot_helper(_params)

    def plot_traces(self):
        """Plot resulting traces of posterior distributions."""
        if self.trace is None:
            raise ValueError("No sampling done!")
        # Plot traces with overlaid means and values
        summary = az.summary(self.trace, stat_funcs={"mean": np.mean}, extend=False)
        ax = az.plot_trace(
            self.trace,
            lines=tuple([(k, {}, v["mean"]) for k, v in summary.iterrows()]),
        )

        for i, mn in enumerate(summary["mean"].values):
            ax[i, 0].annotate(
                f"{mn:.2f}",
                xy=(mn, 0),
                xycoords="data",
                xytext=(5, 10),
                textcoords="offset points",
                rotation=90,
                va="bottom",
                fontsize=16,
                color="C0",
            )

    def evaluate_model(self, method="waic"):
        """Compute WAIC for Bayesian model comparison."""
        if self.trace is None:
            raise ValueError("No sampling done!")
        if method == "waic":
            print(az.waic(self.trace, scale="deviance"))
        elif method == "loo":
            print(az.loo(self.trace, scale="deviance"))
        else:
            raise ValueError("Unknown method!")

    def visualize_model(self):
        """Render visualization of pymc model."""
        if not self.init:
            self.init_model()
        graph = model_to_graphviz(self.model)
        graph.render("graphname", format="png")

    @staticmethod
    def _fun(x, params, backend="numpy"):
        """Define regression function."""
        if backend == "numpy" or backend == "pymc":
            return params[1] * x + params[0]
        else:
            raise ValueError("Unknown backend!")

    @staticmethod
    def _mean_params(bs):
        """Compute mean of parameter traces."""
        bs_mean = ()
        for b in bs:
            bs_mean += (np.mean(b),)
        return bs_mean

    def _plot_helper(self, params):
        """Helper method to make nice plots."""
        x_data, y_data = self.mean_data
        fig, ax = plt.subplots()
        color = ColMap.hex_colors
        _x = np.linspace(0, 10, 1000)
        for i in range(len(params[1])):
            _params_i = tuple(p[i] for p in params)
            y_prior = self._fun(_x, _params_i, "numpy")
            _ = ax.plot(_x, y_prior, color="gray", alpha=0.05)
        _ = ax.plot(
            _x,
            self._fun(_x, self._mean_params(params), "numpy"),
            color=color[1],
            linestyle="-",
        )
        _ = ax.plot(x_data, y_data, color="black", linestyle="--")
        _ = ax.set_xlabel(r"GM/WM $\rightarrow$ GM/CSF")
        _ = ax.set_ylabel("Accuracy in %")
        plt.show()

    @classmethod
    def from_data(cls, sess):
        """Read my data from disk. Data from both sessions are averaged."""
        x = {"0": [], "1": []}
        y = {"0": [], "1": []}
        group = {"0": [], "1": []}
        for i, subj in enumerate(SUBJECTS):
            for day in [0, 1]:
                path = Path(DIR_DATA) / subj / f"{sess}{SESSION[subj][sess][day]}"
                file = path / "v1_bandpass_none" / "accuracy.csv"
                data = np.genfromtxt(file, delimiter=",")
                data = np.mean(data, axis=1)

                x[str(day)].extend(np.arange(N_LAYER))
                y[str(day)].extend(data * 100)
                group[str(day)].extend(i * np.ones_like(data, dtype=np.int64))
        x_mean = (np.array(x["0"]) + np.array(x["1"])) / 2
        y_mean = (np.array(y["0"]) + np.array(y["1"])) / 2
        group_mean = (np.array(group["0"]) + np.array(group["1"])) / 2
        group_mean = np.array(group_mean, dtype=np.int64)

        return cls(x_mean, y_mean, group_mean)


class SineModel(LinearModel):
    """Regression model of a sine function using Bayesian inference.

    Parameters
    ----------
    x : (N,) np.ndarray
        Array of cortical layers.
    y : (N,) np.ndarray
        Array of corresponding decoding accuracies in %.
    group : (N,) np.ndarray
        Array of corresponding integers indicating from which subject the data was
        pooled. Note that in the current version, this parameter is neglected, since
        a pooled Bayesian model is applied.

    """

    def init_model(self):
        """Initialize pymc model. This method therefore contains the prior settings."""
        with self.model:
            b0 = pm.Normal("b0", mu=0, sigma=100)  # interception
            b1 = pm.Normal("b1", mu=np.pi / 2, sigma=np.pi / 8)  # lag
            b2 = pm.Normal("b2", mu=0.2, sigma=0.01)  # frequency
            b3 = pm.Normal("b3", mu=0, sigma=100)  # amplitude

            # define sine model
            yest = self._fun(self.x, (b0, b1, b2, b3), "pymc")

            # define Normal likelihood with HalfCauchy noise (fat tails, equiv to HalfT 1DoF)
            y_sigma = pm.HalfCauchy("y_sigma", beta=10)
            likelihood = pm.Normal(  # noqa: F841
                "likelihood", mu=yest, sigma=y_sigma, observed=self.y
            )

        self.init = True

    def plot_prior(self):
        """Plot predicitive priors."""
        if not self.init:
            self.init_model()
        with self.model:
            prior_checks = pm.sample_prior_predictive(samples=500)
        b0 = prior_checks.prior.b0.to_numpy()[0]
        b1 = prior_checks.prior.b1.to_numpy()[0]
        b2 = prior_checks.prior.b2.to_numpy()[0]
        b3 = prior_checks.prior.b3.to_numpy()[0]
        _params = (b0, b1, b2, b3)
        self._plot_helper(_params)

    def plot_fit(self):
        """Plot resulting data fit."""
        if self.trace is None:
            raise ValueError("No sampling done!")
        b0 = np.mean(self.trace.posterior.b0.to_numpy(), axis=0)
        b1 = np.mean(self.trace.posterior.b1.to_numpy(), axis=0)
        b2 = np.mean(self.trace.posterior.b2.to_numpy(), axis=0)
        b3 = np.mean(self.trace.posterior.b3.to_numpy(), axis=0)
        _params = (b0, b1, b2, b3)
        self._plot_helper(_params)

    @staticmethod
    def _fun(x, params, backend="numpy"):
        """Define regression function."""
        if backend == "numpy":
            return (
                params[3] * np.sin(2 * np.pi * params[2] * (x - params[1])) + params[0]
            )
        elif backend == "pymc":
            return (
                params[3] * pm.math.sin(2 * np.pi * params[2] * (x - params[1]))
                + params[0]
            )
        else:
            raise ValueError("Unknown backend!")
