# -*- coding: utf-8 -*-
"""Spherical mesh module."""

import numpy as np

__all__ = ["Sphere"]


class Sphere:
    """
    Triangle mesh of icosahedron.
    Computation of a spherical mesh based on an icosahedron as primitive. The
    icosahedron consists of 20 equilateral triangles. The primitive can be
    further refined to a sphere by subdivision of triangles. The center point of
    the sphere is in the origin of the coordinate system. The code for
    subdivision is largely taken from [1]_.

    Parameters
    ----------
    scale : float
        Radius of sphere in px.

    Attributes
    ----------
    subdiv : int
        Number of triangle subdivisions.
    middle_point_cache : dict
        Cache to prevent creation of duplicated vertices.
    vtx : (N,3) np.ndarray
        Array of vertex coordinates.
    fac : (N,3) np.ndarray
        Array of corresponding faces.

    References
    ----------
    .. [1] https://sinestesia.co/blog/tutorials/python-icospheres/

    """

    def __init__(self, scale=1):
        self.scale = scale
        self.subdiv = 0
        self.arr_f = None
        self.middle_point_cache = {}

        t = (1 + np.sqrt(5)) / 2  # golden ratio
        self.vtx = np.array(
            [
                self._vertex(-1, t, 0),
                self._vertex(1, t, 0),
                self._vertex(-1, -t, 0),
                self._vertex(1, -t, 0),
                self._vertex(0, -1, t),
                self._vertex(0, 1, t),
                self._vertex(0, -1, -t),
                self._vertex(0, 1, -t),
                self._vertex(t, 0, -1),
                self._vertex(t, 0, 1),
                self._vertex(-t, 0, -1),
                self._vertex(-t, 0, 1),
            ]
        )

        self.fac = np.array(
            [
                # 5 faces around point 0
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                # adjacent faces
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                # 5 faces around 3
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                # adjacent faces
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ]
        )

    def subdivide(self, subdiv):
        """Subdivide icosahedron by splitting each triangle into 4 smaller
        triangles.

        Parameters
        ----------
        subdiv : int
            Number of subdivision iterations.
        """
        self.subdiv += subdiv
        self.middle_point_cache = {}

        for i in range(self.subdiv):
            faces_subdiv = []

            for tri in self.fac:
                v1 = self._middle_point(tri[0], tri[1])
                v2 = self._middle_point(tri[1], tri[2])
                v3 = self._middle_point(tri[2], tri[0])

                faces_subdiv.append([tri[0], v1, v3])
                faces_subdiv.append([tri[1], v2, v1])
                faces_subdiv.append([tri[2], v3, v2])
                faces_subdiv.append([v1, v2, v3])

            self.fac = np.array(faces_subdiv)

    def _vertex(self, x, y, z):
        """Normalize vertex coordinates and scale.

        Parameters
        ----------
        x : float
            x-coordinate.
        y : float
            y-coordinate.
        z : float
            z-coordinate.

        Returns
        -------
        (3,) list
            Scaled coordinates.
        """
        length = np.sqrt(x**2 + y**2 + z**2)

        return [(x * self.scale) / length for x in (x, y, z)]

    def _middle_point(self, ind1, ind2):
        """Find a middle point between two vertices and project to unit sphere.

        Parameters
        ----------
        ind1 : int
            Index of vertex 1.
        ind2 : int
            Index of vertex 2.

        Returns
        -------
        index : int
            Index of created middle point.
        """
        # We check if we have already cut this edge first to avoid duplicated
        # vertices
        smaller_index = min(ind1, ind2)
        greater_index = max(ind1, ind2)

        key = "{0}-{1}".format(smaller_index, greater_index)

        if key in self.middle_point_cache:
            return self.middle_point_cache[key]

        # If it's not in cache, then we can cut it
        vert_1 = self.vtx[ind1, :]
        vert_2 = self.vtx[ind2, :]

        middle = np.mean([vert_1, vert_2], axis=0)
        self.vtx = np.vstack((self.vtx, self._vertex(*middle)))

        index = len(self.vtx) - 1
        self.middle_point_cache[key] = index

        return index

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, s):
        self._scale = s
