# -*- coding: utf-8 -*-
"""Functions for hypothesis testing."""

import numpy as np


def bootstrap(x, Nboot=1000, statfun=np.mean):
    """Calculate bootstrap statistics for a sample x"""
    x = np.array(x)

    resampled_stat = []
    for _ in range(Nboot):
        index = np.random.randint(0, len(x), len(x))  # with replacement
        sample = x[index]
        resampled_stat.append(statfun(sample))

    return resampled_stat


data = np.random.normal(loc=1.0, scale=0.5, size=1000)

bla = bootstrap(data)


import matplotlib.pyplot as plt

plt.hist(bla)
plt.show()


# https://www.gs.washington.edu/academics/courses/akey/56008/lecture/lecture10.pdf
# FDR correction for multiple testing
# Benjamini and Hochberg BH procedure

# control FDR at level delta:
# 1. order the unadjusted p-values: p1 <= p2 <= <= ... <= pm
# 2. then find the test with the highest rank, j, for which the p value, pj, is less than or equal to (j/m)*delta
# 3. declas the tests of rank 1, 2, ..., j as significant
# p(j) <= delta * j/m

# bootstrapping

# confidence interval
# interval that covers 95% of the bootstrapped means (middle 95%)
# to do that, we use the 97.5th percentile and the 2.5th percentile (97.5-2.5=95). In
# other words, if we order all sample means from low to high, and then chop off the
# lowest 2.5% of the means, the middle 95% of the means remain. That range is our
# bootstrapped confidence interval!
# reject null hypothesis if it does not contain the hypothesizes value of the parameter
# 0 (or in our case 50)

# standard error
# standard deviation of bootstrapped distribution

# p value
# https://www.youtube.com/watch?v=N4ZQQqyIf6k&t=308s&ab_channel=StatQuestwithJoshStarmer
# take distribution
# demean (so that mean is zero) -> true null hypothesis
# start bootstrapping
# two-tailed p-value: counter data >= and <= mean
# compare counted val to 0.05
# if >= 0.05 -> fail rejection
data_demeaned = data - np.mean(data)
data_bootstrap = bootstrap(data_demeaned)
