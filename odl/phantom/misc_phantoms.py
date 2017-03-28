﻿# Copyright 2014-2016 The ODL development group
#
# This file is part of ODL.
#
# ODL is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ODL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ODL.  If not, see <http://www.gnu.org/licenses/>.

"""Miscellaneous phantoms that do not fit in other categories."""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
standard_library.install_aliases()

import numpy as np


__all__ = ('submarine', 'disc_phantom', 'donut', 'sphere',
           'sphere2', 'cube', 'particles_3d')


def submarine(space, smooth=True, taper=20.0):
    """Return a 'submarine' phantom consisting in an ellipsoid and a box.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
    smooth : bool, optional
        If ``True``, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : float, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : ``space`` element
        The submarine phantom in ``space``.
    """
    if space.ndim == 2:
        if smooth:
            return _submarine_2d_smooth(space, taper)
        else:
            return _submarine_2d_nonsmooth(space)
    else:
        raise ValueError('phantom only defined in 2 dimensions, got {}'
                         ''.format(space.ndim))


def _submarine_2d_smooth(space, taper):
    """Return a 2d smooth 'submarine' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_ellipse(x):
        """Blurred characteristic function of an ellipse.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the ellipse is centered at ``(0.6, 0.3)`` and has half-axes
        ``(0.4, 0.14)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent()
        center = np.array([0.6, 0.3]) * space.domain.extent()
        center += space.domain.min()

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    def blurred_rect(x):
        """Blurred characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the rect has lower left ``(0.56, 0.4)`` and upper right
        ``(0.76, 0.6)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.56, 0.4]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            length = upp - low
            out = out * (logistic((xi - low) / length, taper) *
                         logistic((upp - xi) / length, taper))
        return out

    out = space.element(blurred_ellipse)
    out += space.element(blurred_rect)
    return out.ufuncs.minimum(1, out=out)


def _submarine_2d_nonsmooth(space):
    """Return a 2d nonsmooth 'submarine' phantom."""

    def ellipse(x):
        """Characteristic function of an ellipse.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the ellipse is centered at ``(0.6, 0.3)`` and has half-axes
        ``(0.4, 0.14)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.14]) * space.domain.extent()
        center = np.array([0.6, 0.3]) * space.domain.extent()
        center += space.domain.min()

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    def rect(x):
        """Characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1]``,
        the rect has lower left ``(0.56, 0.4)`` and upper right
        ``(0.76, 0.6)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.56, 0.4]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.76, 0.6]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            out = out * ((xi >= low) & (xi <= upp))
        return out

    out = space.element(ellipse)
    out += space.element(rect)
    return out.ufuncs.minimum(1, out=out)


def disc_phantom(discr, smooth=True, taper=20.0):
    """Return a 'disc' phantom.

    This phantom is used in [Okt2015]_ for shape-based reconstruction.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpVector`
    """
    if discr.ndim == 2:
        if smooth:
            return _disc_phantom_2d_smooth(discr, taper)
        else:
            return _disc_phantom_2d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 2 dimensions, got {}.'
                         ''.format(discr.dim))


def _disc_phantom_2d_smooth(discr, taper):
    """Return a 2d smooth 'disc' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_circle(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.2, 0.2)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_circle)
    return out.ufuncs.minimum(1, out=out)


def _disc_phantom_2d_nonsmooth(discr):
    """Return a 2d nonsmooth 'disc' phantom."""

    def circle(x):
        """Characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.2, 0.2)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(circle)
    return out.ufuncs.minimum(1, out=out)


def donut(discr, smooth=True, taper=20.0):
    """Return a 'donut' phantom.

    This phantom is used in [Okt2015]_ for shape-based reconstruction.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 2:
        if smooth:
            return _donut_2d_smooth(discr, taper)
        else:
            return _donut_2d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 2 dimensions, got {}.'
                         ''.format(discr.dim))


def _donut_2d_smooth(discr, taper):
    """Return a 2d smooth 'donut' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_circle_1(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.4]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    def blurred_circle_2(x):
        """Blurred characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_circle_1) - discr.element(blurred_circle_2)
    return out.ufuncs.minimum(1, out=out)


def _donut_2d_nonsmooth(discr):
    """Return a 2d nonsmooth 'donut' phantom."""

    def circle_1(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.4, 0.4]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    def circle_2(x):
        """Characteristic function of an circle.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0)`` and has half-axes
        ``(0.25, 0.25)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.2, 0.2]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(circle_1) - discr.element(circle_2)
    
    return out.ufuncs.minimum(1, out=out)


def sphere(discr, smooth=True, taper=20.0):
    """Return a 'sphere' phantom.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 3:
        if smooth:
            return _sphere_3d_smooth(discr, taper)
        else:
            return _sphere_3d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 3 dimensions, got {}.'
                         ''.format(discr.dim))


def _sphere_3d_smooth(discr, taper):
    """Return a 3d smooth 'sphere' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_sphere(x):
        """Blurred characteristic function of a sphere.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.1, 0.1, 0.1]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_sphere)
    return out.ufuncs.minimum(1, out=out)


def _sphere_3d_nonsmooth(discr):
    """Return a 3d nonsmooth 'sphere' phantom."""

    def sphere(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.1, 0.1, 0.1]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(sphere)
    return out.ufuncs.minimum(1, out=out)


def sphere2(discr, smooth=True, taper=20.0):
    """Return a 'sphere' phantom.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 3:
        if smooth:
            return _sphere_3d_smooth2(discr, taper)
        else:
            return _sphere_3d_nonsmooth2(discr)
    else:
        raise ValueError('Phantom only defined in 3 dimensions, got {}.'
                         ''.format(discr.dim))


def _sphere_3d_smooth2(discr, taper):
    """Return a 3d smooth 'sphere' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_sphere(x):
        """Blurred characteristic function of a sphere.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, -0.5, 0.0]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_sphere)
    return out.ufuncs.minimum(1, out=out)


def _sphere_3d_nonsmooth2(discr):
    """Return a 3d nonsmooth 'sphere' phantom."""

    def sphere(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, -0.5, 0.0]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(sphere)
    return out.ufuncs.minimum(1, out=out)


def cube(space, smooth=True, taper=20.0):
    """Return a 3D 'cube' phantom.

    Parameters
    ----------
    space : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created.
    smooth : bool, optional
        If ``True``, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : float, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : ``space`` element
        The submarine phantom in ``space``.
    """
    if space.ndim == 3:
        if smooth:
            return _cube_3d_smooth(space, taper)
        else:
            return _cube_3d_nonsmooth(space)
    else:
        raise ValueError('phantom only defined in 3 dimensions, got {}'
                         ''.format(space.ndim))


def _cube_3d_smooth(space, taper):
    """Return a 2d smooth 'cube' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_cube(x):
        """Blurred characteristic function of a cube.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1] x [0, 1]``,
        the rect has lower left ``(0.35, 0.35, 0.35)`` and upper right
        ``(0.65, 0.65, 0.65)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.35, 0.35, 0.35]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.65, 0.65, 0.65]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            length = upp - low
            out = out * (logistic((xi - low) / length, taper) *
                         logistic((upp - xi) / length, taper))
        return out

    out = space.element(blurred_cube)
    return out.ufuncs.minimum(1, out=out)


def _cube_3d_nonsmooth(space):
    """Return a 2d nonsmooth 'cube' phantom."""

    def cube(x):
        """Characteristic function of a rectangle.

        If ``space.domain`` is a rectangle ``[0, 1] x [0, 1] x [0, 1]``,
        the rect has lower left ``(0.35, 0.35, 0.35)`` and upper right
        ``(0.65, 0.65, 0.65)``. For other domains, the values are scaled
        accordingly.
        """
        xlower = np.array([0.35, 0.35, 0.35]) * space.domain.extent()
        xlower += space.domain.min()
        xupper = np.array([0.65, 0.65, 0.65]) * space.domain.extent()
        xupper += space.domain.min()

        out = np.ones_like(x[0])
        for xi, low, upp in zip(x, xlower, xupper):
            out = out * ((xi >= low) & (xi <= upp))
        return out

    out = space.element(cube)
    return out.ufuncs.minimum(1, out=out)


def particles_3d(discr, smooth=True, taper=20.0):
    """Return a 'two particles' phantom.

    Parameters
    ----------
    discr : `DiscreteLp`
        Discretized space in which the phantom is supposed to be created
    smooth : `bool`, optional
        If `True`, the boundaries are smoothed out. Otherwise, the
        function steps from 0 to 1 at the boundaries.
    taper : `float`, optional
        Tapering parameter for the boundary smoothing. Larger values
        mean faster taper, i.e. sharper boundaries.

    Returns
    -------
    phantom : `DiscreteLpElement`
    """
    if discr.ndim == 3:
        if smooth:
            return _particles_3d_smooth(discr, taper)
        else:
            return _particles_3d_nonsmooth(discr)
    else:
        raise ValueError('Phantom only defined in 3 dimensions, got {}.'
                         ''.format(discr.dim))


def _particles_3d_smooth(discr, taper):
    """Return a 3d smooth 'sphere' phantom."""

    def logistic(x, c):
        """Smoothed step function from 0 to 1, centered at 0."""
        return 1. / (1 + np.exp(-c * x))

    def blurred_sphere1(x):
        """Blurred characteristic function of a sphere.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, 0.4]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)
    
    def blurred_sphere2(x):
        """Blurred characteristic function of a sphere.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.1, -0.4]) * discr.domain.extent() / 2

        # Efficiently calculate |z|^2, z = (x - center) / radii
        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        out = np.sqrt(sq_ndist)
        out -= 1
        # Return logistic(taper * (1 - |z|))
        return logistic(out, -taper)

    out = discr.element(blurred_sphere1) + discr.element(blurred_sphere2)
    return out.ufuncs.minimum(1, out=out)


def _particles_3d_nonsmooth(discr):
    """Return a 3d nonsmooth 'sphere' phantom."""

    def blurred_sphere1(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.0, 0.4]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)
    
    def blurred_sphere2(x):
        """Characteristic function of an ellipse.
        If ``discr.domain`` is a rectangle ``[-1, 1] x [-1, 1] x [-1, 1]``,
        the circle is centered at ``(0.0, 0.0, 0.0)`` and has half-axes
        ``(0.05, 0.05, 0.05)``. For other domains, the values are scaled
        accordingly.
        """
        halfaxes = np.array([0.05, 0.05, 0.05]) * discr.domain.extent() / 2
        center = np.array([0.0, 0.1, -0.4]) * discr.domain.extent() / 2

        sq_ndist = np.zeros_like(x[0])
        for xi, rad, cen in zip(x, halfaxes, center):
            sq_ndist = sq_ndist + ((xi - cen) / rad) ** 2

        return np.where(sq_ndist <= 1, 1, 0)

    out = discr.element(blurred_sphere1) + discr.element(blurred_sphere2)
    return out.ufuncs.minimum(1, out=out)


if __name__ == '__main__':
    # Show the phantoms
    import odl

    space = odl.uniform_discr([-1, -1], [1, 1], [300, 300])
    submarine(space, smooth=False).show('submarine smooth=False')
    submarine(space, smooth=True).show('submarine smooth=True')
    submarine(space, smooth=True, taper=50).show('submarine taper=50')

    disc_phantom(space, smooth=False).show('disc smooth=False')
    disc_phantom(space, smooth=True).show('disc smooth=True')
    disc_phantom(space, smooth=True, taper=50).show('disc taper=50')

    donut(space, smooth=False).show('donut smooth=False')
    donut(space, smooth=True).show('donut smooth=True')
    donut(space, smooth=True, taper=50).show('donut taper=50')
    
    space_3d = odl.uniform_discr([-1, -1, -1], [1, 1, 1], [300, 300, 300])
    
    sphere(space_3d, smooth=False).show('sphere smooth=False',
          indices=np.s_[:, :, space_3d.shape[-1] // 2])
    sphere(space_3d, smooth=True).show('sphere smooth=True',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    sphere(space_3d, smooth=True, taper=50).show('sphere taper=50',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    
    sphere2(space_3d, smooth=False).show('sphere smooth=False',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    sphere2(space_3d, smooth=True).show('sphere smooth=True',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    sphere2(space_3d, smooth=True, taper=50).show('sphere taper=50',
          indices=np.s_[space_3d.shape[-1] // 2, :, :])
    
    cube(space_3d, smooth=False).show('cube smooth=False',
        indices=np.s_[space_3d.shape[-1] // 2, :, :])
    cube(space_3d, smooth=True).show('cube smooth=True',
        indices=np.s_[space_3d.shape[-1] // 2, :, :])
    cube(space_3d, smooth=True, taper=50).show('cube taper=50',
        indices=np.s_[space_3d.shape[-1] // 2, :, :])

    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
