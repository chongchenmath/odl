# Copyright 2014-2016 The ODL development group
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

"""
Example of shape-based image reconstruction
using optimal information transformation.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.util import snr
from odl.deform import optimal_information_transport_solver
import odl
standard_library.install_aliases()

# Give input images
I0name = './pictures/c_highres.png'
I1name = './pictures/i_highres.png'
# I0name = './pictures/handnew1.png'
# I1name = './pictures/DS0002AxialSlice80.png'

# Get digital images
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]

# Discrete reconstruction space: discretized functions on the rectangle
space = odl.uniform_discr(
    min_corner=[-16, -16], max_corner=[16, 16], nsamples=[128, 128],
    dtype='float32', interp='linear')

# Give the number of directions
num_angles = 6

# Create the uniformly distributed directions
angle_partition = odl.uniform_partition(0, np.pi, num_angles,
                                        nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = odl.uniform_partition(-24, 24, 192)

# Create 2-D parallel projection geometry
geometry = odl.tomo.Parallel2dGeometry(angle_partition,
                                       detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create the ground truth as the given image
ground_truth = space.element(I0)

# # Create the ground truth as the submarine phantom
# ground_truth = odl.util.submarine_phantom(space, smooth=True, taper=50.0)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = odl.util.white_noise(ray_trafo.range) * 1.0

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Compute the signal-to-noise ratio in dB
snr = snr(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Maximum iteration number
niter = 2000

callback = odl.solvers.CallbackShow(
    'iterates', display_step=50) & odl.solvers.CallbackPrintIteration()

# Create the template as the given image
template = space.element(I1)

# # Create the template as the disc phantom
# template = odl.util.disc_phantom(space, smooth=True, taper=50.0)

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'nmp', 'mp' means mass-preserving method,
# 'nmp' means non-mass-preserving method
impl1 = 'nmp'

# Implementation method for image matching or image reconstruction,
# impl chooses 'matching' or 'reconstruction', 'matching' means image matching,
# 'reconstruction' means image reconstruction
impl2 = 'matching'

# Normalize the template's density as the same as the ground truth if consider
# mass preserving method
if impl1 == 'mp':
    template *= np.sum(ground_truth) / np.sum(template)

ground_truth.show('phantom')
template.show('template')

# For image reconstruction
if impl2 == 'reconstruction':
    # Give step size for solver
    eps = 0.0002

    # Give regularization parameter
    sigma = 1e3

    # Create the forward operator for image reconstruction
    op = ray_trafo

    # Create the gradient operator for the L2 functional
    gradS = op.adjoint * odl.ResidualOperator(op, noise_proj_data)

    # Compute by optimal information transport solver
    optimal_information_transport_solver(gradS, template, niter,
                                         eps, sigma, impl1, callback)

# For image matching
if impl2 == 'matching':
    # Give step size for solver
    eps = 0.02

    # Give regularization parameter
    sigma = 1

    # Create the forward operator for image matching
    op = odl.IdentityOperator(space)

    # Create the gradient operator for the L2 functional
    gradS = op.adjoint * odl.ResidualOperator(op, ground_truth)

    # Compute by optimal information transport solver
    optimal_information_transport_solver(gradS, template, niter,
                                         eps, sigma, impl1, callback)
