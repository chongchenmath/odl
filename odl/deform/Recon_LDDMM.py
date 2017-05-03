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
Image reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.discr import uniform_discr, uniform_partition
from odl.tomo import Parallel2dGeometry, RayTransform, fbp_op
from odl.solvers import CallbackShow, CallbackPrintIteration
from odl.phantom import (white_noise, disc_phantom, submarine,
                         shepp_logan, geometric)
from odl.deform.LDDMM_gradiant_descent_scheme import (
        LDDMM_gradient_descent_solver)
standard_library.install_aliases()


def shepp_logan_ellipse_2d_template():
    """Return ellipse parameters for a 2d Shepp-Logan phantom.

    This assumes that the ellipses are contained in the square
    [-1, -1]x[-1, -1].
    """
#    return [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
#            [-.98, .6624, .8740, 0.0000, -.0184, 0],
#            [-.02, .1100, .3100, 0.2200, 0.0000, -18],
#            [-.02, .1600, .4100, -.2200, 0.0000, 18],
#            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
#            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
#            [0.01, .0460, .0460, 0.0000, -.1000, 0],
#            [0.01, .0460, .0230, -.0800, -.6050, 0],
#            [0.01, .0230, .0230, 0.0000, -.6060, 0],
#            [0.01, .0230, .0460, 0.0600, -.6050, 0]]
    #       value  axisx  axisy     x       y  rotation           
    # Shepp-Logan region of interest
    return [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.02, .1400, .1400, 0.2200, 0.0000, -18],
            [-.02, .1600, .4100, -.2200, 0.0000, 18],
            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
            [0.01, .0460, .0460, 0.0000, -.1000, 0],
            [0.01, .0460, .0230, -.0800, -.6050, 0],
            [0.01, .0230, .0230, 0.0000, -.6060, 0],
            [0.01, .0230, .0460, 0.0600, -.6050, 0]]
#    return [[2.00, .6000, .6000, 0.0000, 0.1200, 0],
#            [-.98, .5624, .5640, 0.0000, -.0184 + 0.12, 0],
#            [-.02, .1100, .1100, 0.2600, 0.1500, -18],
#            [-.02, .1300, .1300, -.2500, 0.2000, 18],
#            [0.01, .1650, .1650, 0.0000, 0.3000, 0],
#            [0.01, .0300, .0300, 0.0000, 0.1400, 0],
#            [0.01, .0300, .0300, -.1400, 0.1000, 0],
#            [0.01, .0360, .0230, -.0770, -.2050, 0],
#            [0.01, .0230, .0230, 0.0000, -.2060, 0],
#            [0.01, .0230, .0360, 0.0600, -.2050, 0]] 

#template = shepp_logan_2d(space, modified=True)
#template.show('template')


def modified_shepp_logan_ellipses(ellipses):
    """Modify ellipses to give the modified Shepp-Logan phantom.

    Works for both 2d and 3d.
    """
    intensities = [1.0, -0.8, -0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    assert len(ellipses) == len(intensities)

    for ellipse, intensity in zip(ellipses, intensities):
        ellipse[0] = intensity


def shepp_logan_ellipses(ndim, modified=False):
    """Ellipses for the standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    ndim : {2, 3}
        Dimension of the space the ellipses should be in.
    modified : bool, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    ellipse_phantom : Function for creating arbitrary ellipse phantoms
    shepp_logan : Create a phantom with these ellipses
    """
    if ndim == 2:
        ellipses = shepp_logan_ellipse_2d_template()
    else:
        raise ValueError('dimension not 2, no phantom available')

    if modified:
        modified_shepp_logan_ellipses(ellipses)

    return ellipses


def shepp_logan_2d(space, modified=False):
    """Standard `Shepp-Logan phantom`_ in 2 or 3 dimensions.

    Parameters
    ----------
    space : `DiscreteLp`
        Space in which the phantom is created, must be 2- or 3-dimensional.
    modified : `bool`, optional
        True if the modified Shepp-Logan phantom should be given.
        The modified phantom has greatly amplified contrast to aid
        visualization.

    See Also
    --------
    shepp_logan_ellipses : Get the parameters that define this phantom
    ellipse_phantom : Function for creating arbitrary ellipse phantoms
    """
    ellipses = shepp_logan_ellipses(space.ndim, modified)

    return geometric.ellipse_phantom(space, ellipses)


def snr(signal, noise, impl):
    """Compute the signal-to-noise ratio.
    Parameters
    ----------
    signal : `array-like`
        Noiseless data.
    noise : `array-like`
        Noise.
    impl : {'general', 'dB'}
        Implementation method.
        'general' means SNR = variance(signal) / variance(noise),
        'dB' means SNR = 10 * log10 (variance(signal) / variance(noise)).
    Returns
    -------
    snr : `float`
        Value of signal-to-noise ratio.
        If the power of noise is zero, then the return is 'inf',
        otherwise, the computed value.
    """
    if np.abs(np.asarray(noise)).sum() != 0:
        ave1 = np.sum(signal) / signal.size
        ave2 = np.sum(noise) / noise.size
        s_power = np.sqrt(np.sum((signal - ave1) * (signal - ave1)))
        n_power = np.sqrt(np.sum((noise - ave2) * (noise - ave2)))
        if impl == 'general':
            return s_power / n_power
        elif impl == 'dB':
            return 10.0 * np.log10(s_power / n_power)
        else:
            raise ValueError('unknown `impl` {}'.format(impl))
    else:
        return float('inf')


# --- Give input images --- #

#I0name = './pictures/c_highres.png'
#I1name = './pictures/i_highres.png'
#I0name = './pictures/DS0003AxialSlice80.png' # 256 * 256, I0[:,:,1]
#I1name = './pictures/DS0002AxialSlice80.png'
#I0name = './pictures/hand5.png'
#I1name = './pictures/hand3.png'
#I0name = './pictures/handnew1.png'
#I1name = './pictures/handnew2.png'
#I0name = './pictures/v.png' # 64 * 64
#I1name = './pictures/j.png' # 64 * 64
#I0name = './pictures/ImageHalf058.png'
#I1name = './pictures/ImageHalf059.png'
#I0name = './pictures/ImageHalf068.png'
#I1name = './pictures/ImageHalf069.png'
I0name = './pictures/ss_save.png' # 438 * 438, I0[:,:,1]
I1name = './pictures/ss_save_1.png' # 438 * 438, I0[:,:,1]

# --- Get digital images --- #

#I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[::2, ::2]
#I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[::2, ::2]
I0 = np.rot90(plt.imread(I0name).astype('float'), -1)[:,:,1]
I1 = np.rot90(plt.imread(I1name).astype('float'), -1)[:,:,1]

# Discrete reconstruction space: discretized functions on the rectangle
rec_space = uniform_discr(
    min_pt=[-16, -16], max_pt=[16, 16], shape=[438, 438],
    dtype='float32', interp='linear')

# Create the ground truth as the given image
ground_truth = rec_space.element(I0)

# Create the ground truth as the submarine phantom
# ground_truth = submarine_phantom(space, smooth=True, taper=50.0)

# Create the ground truth as the Shepp-Logan phantom
# ground_truth = shepp_logan(space, modified=True)

# Create the template as the given image
template = rec_space.element(I1)

# Create the template as the disc phantom
# template = disc_phantom(space, smooth=True, taper=50.0)

# Create the template as the deformed Shepp-Logan phantom
# template = shepp_logan_2d(space, modified=True)

# Create the template for Shepp-Logan phantom
#deform_field_space = space.vector_field_space
#disp_func = [
#    lambda x: 16.0 * np.sin(np.pi * x[0] / 40.0),
#    lambda x: 16.0 * np.sin(np.pi * x[1] / 36.0)]
#deform_field = deform_field_space.element(disp_func)
#template = space.element(geometric_deform(shepp_logan(space, modified=True),
#                                          deform_field))

# Maximum iteration number
niter = 200

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl1 = 'geom'

# Normalize the template's density as the same as the ground truth if consider
# mass preserving method
if impl1 == 'mp':
#    template *= np.sum(ground_truth) / np.sum(template)
    template *= np.linalg.norm(ground_truth, 'fro')/ \
        np.linalg.norm(template, 'fro')

# Implementation method for least square data matching term
impl2 = 'least_square'

# Show intermiddle results
callback = CallbackShow(
    '{!r} {!r} iterates'.format(impl1, impl2), display_step=5) & \
    CallbackPrintIteration()

#ground_truth.show('ground truth')
#template.show('template')

# Give kernel function
def kernel(x):
    sigma = 2.0
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Give step size for solver
eps = 0.05

# Give regularization parameter
lamb = 0.0000001

# Give the number of directions
num_angles = 4

# Create the uniformly distributed directions
angle_partition = uniform_partition(0.0, np.pi, num_angles,
                                    nodes_on_bdry=[(True, False)])

# Create 2-D projection domain
# The length should be 1.5 times of that of the reconstruction space
detector_partition = uniform_partition(-24, 24, 620)

# Create 2-D parallel projection geometry
geometry = Parallel2dGeometry(angle_partition, detector_partition)

# Ray transform aka forward projection. We use ASTRA CUDA backend.
forward_op = RayTransform(rec_space, geometry, impl='astra_cpu')

# Create projection data by calling the op on the phantom
proj_data = forward_op(ground_truth)

# Add white Gaussion noise onto the noiseless data
noise = 1.0 * white_noise(forward_op.range)

# Add white Gaussion noise from file
# noise = op.range.element(np.load('noise_20angles.npy'))

# Create the noisy projection data
noise_proj_data = proj_data + noise

# Create the noisy data from file
#noise_proj_data = op.range.element(
#    np.load('noise_proj_data_20angles_snr_4_98.npy'))

#    # --- Reconstructing by FBP --- #    

#    # Create FBP operator
#    FBP = fbp_op(op, padding=True, filter_type='Hamming',
#                 frequency_scaling=0.8)
#    # Calculate filtered backprojection of data             
#    fbp_reconstruction = FBP(proj_data)
#    
#    # Shows result of FBP reconstruction
#    fbp_reconstruction.show(title='Filtered backprojection')

# Compute the signal-to-noise ratio in dB
snr = snr(proj_data, noise, impl='dB')

# Output the signal-to-noise ratio
print('snr = {!r}'.format(snr))

# Give the number of time points
time_itvs = 20

# Compute by LDDMM solver
image_N0, E = LDDMM_gradient_descent_solver(forward_op, noise_proj_data, template,
                                         time_itvs, niter, eps, lamb,
                                         kernel, impl1, impl2, callback)
    
rec_result_1 = rec_space.element(image_N0[time_itvs // 4])
rec_result_2 = rec_space.element(image_N0[time_itvs // 4 * 2])
rec_result_3 = rec_space.element(image_N0[time_itvs // 4 * 3])
rec_result = rec_space.element(image_N0[time_itvs])

# Compute the projections of the reconstructed image
rec_proj_data = forward_op(rec_result)

#%%%
# Plot the results of interest
plt.figure(1, figsize=(24, 24))
#plt.clf()

plt.subplot(3, 3, 1)
plt.imshow(np.rot90(template), cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max())
plt.colorbar()
plt.title('Template')

plt.subplot(3, 3, 2)
plt.imshow(np.rot90(rec_result_1), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()) 
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4))

plt.subplot(3, 3, 3)
plt.imshow(np.rot90(rec_result_2), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()) 
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 2))

plt.subplot(3, 3, 4)
plt.imshow(np.rot90(rec_result_3), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()) 
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 4 * 3))

plt.subplot(3, 3, 5)
plt.imshow(np.rot90(rec_result), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max()) 
plt.colorbar()
plt.title('Reconstructed by {!r} iters, '
    '{!r} projs'.format(niter, num_angles))

plt.subplot(3, 3, 6)
plt.imshow(np.rot90(ground_truth), cmap='bone',
           vmin=np.asarray(ground_truth).min(),
           vmax=np.asarray(ground_truth).max())
plt.colorbar()
plt.title('Ground truth')

plt.subplot(3, 3, 7)
plt.plot(np.asarray(proj_data)[0], 'b', np.asarray(noise_proj_data)[0],
         'r', np.asarray(rec_proj_data)[0], 'g'), 
plt.axis([0, 619, -4, 13]), plt.grid(True)
#    plt.title('$\Theta=0^\circ$, b: truth, r: noisy, '
#        'g: rec_proj, SNR = {:.3}dB'.format(snr))
#    plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(3, 3, 8)
plt.plot(np.asarray(proj_data)[2], 'b', np.asarray(noise_proj_data)[2],
         'r', np.asarray(rec_proj_data)[2], 'g'),
plt.axis([0, 619, -4, 13]), plt.grid(True)
#    plt.title('$\Theta=90^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])

plt.subplot(3, 3, 9)
plt.plot(np.asarray(proj_data)[4], 'b', np.asarray(noise_proj_data)[4],
         'r', np.asarray(rec_proj_data)[4], 'g'),
plt.axis([0, 619, -4, 13]), plt.grid(True)
#    plt.title('$\Theta=162^\circ$')
#    plt.gca().axes.yaxis.set_ticklabels([])'

plt.figure(2, figsize=(8, 1.5))
#plt.clf()
plt.plot(E)
plt.ylabel('Energy')
# plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
plt.gca().axes.yaxis.set_ticklabels([])
plt.grid(True)


##%%%
## --- Reconstructing by FBP --- #    
##
##
## Create FBP operator
#FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
#             frequency_scaling=0.4)
## Implement FBP method            
#rec_result_FBP = FBP(noise_proj_data)
#
## Show result
#plt.figure(3, figsize=(4, 4))
#plt.imshow(np.rot90(rec_result_FBP), cmap='bone')
#plt.colorbar()
#plt.title('Reconstructed by FBP, {!r} projs'.format(num_angles))
#
#
##%%%
## --- Reconstructing by TV method --- #   
#from odl.operator import (BroadcastOperator, power_method_opnorm)
#from odl.solvers import (CallbackShow, CallbackPrintIteration, ZeroFunctional,
#                         L2NormSquared, L1Norm, SeparableSum, 
#                         chambolle_pock_solver)
#from odl.discr import Gradient
#
## Initialize gradient operator
#grad_op = Gradient(rec_space, method='forward', pad_mode='symmetric')
#
## Column vector of two operators
#op = BroadcastOperator(forward_op, grad_op)
#
## Do not use the g functional, set it to zero.
#g = ZeroFunctional(op.domain)
#
## Set regularization parameter
#lamb = 3.0
#
## Isotropic TV-regularization i.e. the l1-norm
#l1_norm = lamb * L1Norm(grad_op.range)
#
## l2-squared data matching
#l2_norm = L2NormSquared(forward_op.range).translated(noise_proj_data)
#
## --- Select solver parameters and solve using Chambolle-Pock --- #
## Estimate operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
#op_norm = 1.1 * power_method_opnorm(op)
#
#niter = 1000  # Number of iterations
#tau = 1.0 / op_norm  # Step size for the primal variable
#sigma = 1.0 / op_norm  # Step size for the dual variable
#gamma = 0.5
#
## Choose a starting point
#x = forward_op.domain.zero()
#
## Create functionals for the dual variable
## Combine functionals, order must correspond to the operator K
#f = SeparableSum(l2_norm, l1_norm)
#
## Optionally pass callback to the solver to display intermediate results
#callback = (CallbackPrintIteration() &
#            CallbackShow('iterates'))
#
## Run the algorithm
#chambolle_pock_solver(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
#                      gamma=gamma, callback=callback)
#
#plt.figure(4, figsize=(4, 4))
#plt.imshow(np.rot90(x), cmap='bone')
#plt.colorbar()
#plt.title('Reconstructed by TV, {!r} projs'.format(num_angles))
