# -*- coding: utf-8 -*-
"""Functions for hypothesis testing."""

import functools

import numpy as np


class Bootstrap:
    """Bootstrap statistics.

    Parameters
    ----------
    sample : (N,) np.ndarray
        Input distribution.
    Nboot : int
        Number of bootstrapping iterations. The default is 1000.
    statfun : str
        Used statistics. The default is np.mean.

    """

    def __init__(self, sample, Nboot=1000, statfun=np.mean):
        self.sample = np.array(sample)
        self.Nboot = Nboot
        self.statfun = statfun

    @property
    @functools.lru_cache
    def dist(self):
        """Calculate boostrap distribution."""
        return self._bootstrap_dist(self.sample, self.Nboot, self.statfun)

    def confidence_interval(self, level=95):
        """Get confidence interval from bootstrap distribution. The interval covers by
        default the 95% of the bootstrap means (middle 95%). To do that, we use the
        97.5th and the 2.5th percentile (97.5-2.5=95). In other words, if we order all
        sample means from low to high, and then chop off the lowest and highest 2.5%
        of the means, the middle 95% of the means remain. That range is our bootstrapped
        confidence interval. We can reject the null hypothesis if it does not contain
        the hypothesized value of the parameter 0.

        Parameters
        ----------
        level : int
            Confidence interval. The default is 95.
        """
        low = np.percentile(self.dist, (100 - level) / 2)
        high = np.percentile(self.dist, 100 - (100 - level) / 2)
        return low, high

    def p_value(self, n_iter=1000, chance_level=50.0):
        """p-value from bootstrap distribution. We start by taking the distribution and
        remove the mean. This gives us the true null hypothesis. From this distrbution,
        we create then a bootstrapped null hypothesis. We test the mean of our sample
        (after subtraction of the chance level) to the bootstrapped null hypothesis to
        get a two-tailed p-value. [1]_

        Parameters
        ----------
        n_iter : int
            Number of bootstrapping iterations. The default is 1000.
        chance_level : float
            Chance level that which we test. The default is 50.0.

        References
        ----------
        .. [1] https://www.youtube.com/watch?v=N4ZQQqyIf6k&t=308s&ab_channel=StatQuestwithJoshStarmer
        """
        null = self.sample - np.mean(self.sample)
        resampled_null = self._bootstrap_dist(null, n_iter, self.statfun)
        statistic = self.statfun(self.sample) - chance_level
        low = sum(i <= -statistic for i in resampled_null)
        high = sum(i >= statistic for i in resampled_null)
        return (low + high) / n_iter

    @staticmethod
    def _bootstrap_dist(x, n, func):
        """Generate bootstrapped distirbution

        Parameters
        ----------
        x : (N,) np.ndarray
            Input distribution.
        n : int
            Number of bootstrapping iterations.
        func : str
            Used statistics.
        """
        resampled_stat = []
        for _ in range(n):
            index = np.random.randint(0, len(x), len(x))  # with replacement
            sample = x[index]
            resampled_stat.append(func(sample))
        return resampled_stat


def fdr_correction(arr_p):
    """FDR correction for multiple testing. This function computes adjusted p-values
    using the Benhamini and Hochberg (BH) procedure. This is nicely explained in [1]_
    (see [2]_ for more information about FDR correction). For the procedure, p-values
    are sorted (ranked) based on their value in ascending order k = 1..m. Then, adjusted
    p-values are computed by p(k) * m / k. Finally, we loop backwards through the
    adjusted p-value array and and equalize all p-values with their predecessor if they
    are they larger than their predecessor.

    References
    ----------
    .. [1] https://youtu.be/rZKa4tW2NKs
    .. [2] https://www.gs.washington.edu/academics/courses/akey/56008/lecture/lecture10.pdf
    """
    arr_p = np.array(arr_p)
    index = np.argsort(arr_p)
    arr_p_adj = arr_p * len(arr_p) / (index + 1)

    for i in range(len(arr_p) - 2, -1, -1):
        p_curr = arr_p_adj[index[i]]
        p_after = arr_p_adj[index[i + 1]]
        if p_curr > p_after:
            arr_p_adj[index[i]] = p_after

    return arr_p_adj
