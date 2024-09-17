# -*- coding: utf-8 -*-
"""
Main period of Laplacian Gaussian.

This script calculates the filter sizes of the bandpass filter for the figure, which
illustrates decoding profiles with different filter sizes.
"""

import os
import sys

from cortex.polyutils import Surface
from fmri_tools.surface.filter import LaplacianGaussian

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.sphere import Sphere

# constants
SUBDIV = 4  # number of triangle subdivisions
SUBDIV_ITER = 4  # number of subdivision iterations
N_ITER = 100  # number of iterations to estimate filter scale
T = [0.025, 0.05, 0.1, 0.5]  # filter sizes


def _main():
    sphere = Sphere(SUBDIV)#SphereMesh(SUBDIV)
    sphere.subdivide(SUBDIV_ITER)
    surf = Surface(sphere.vtx, sphere.fac)

    print(f"Number of vertices: {len(sphere.vtx)}")
    print(f"Average edge length: {surf.avg_edge_length}")
    for t in T:
        filt = LaplacianGaussian(sphere.vtx, sphere.fac, t)
        res = filt.spatial_scale(n_iter=N_ITER)
        print(f"Main filter period for t={t}: {res['period']}")


if __name__ == "__main__":
    _main()
