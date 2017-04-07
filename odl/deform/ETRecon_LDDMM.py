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
ET image reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.discr import uniform_discr, Gradient
from odl.phantom import sphere, sphere2, cube, particles_3d
from odl.deform.LDDMM_gradiant_descent_scheme import (
        LDDMM_gradient_descent_solver)
from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
from odl.tomo import RayTransform, fbp_op
from odl.operator import (BroadcastOperator, power_method_opnorm)
from odl.solvers import (CallbackShow, CallbackPrintIteration, ZeroFunctional,
                         L2NormSquared, L1Norm, SeparableSum, 
                         chambolle_pock_solver, conjugate_gradient_normal)
standard_library.install_aliases()


#%%%
# --- Reading data --- #

# Get the path of data
directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
data_filename = 'triangle_crop.mrc'
file_path = directory + data_filename
data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
                                                           force_type='FEI1',
                                                           normalize=True)

#Downsample the data
downsam = 1
data_downsam = data[:, :, ::downsam]

# --- Getting geometry --- #
det_pix_size = 0.521

# Create 3-D parallel projection geometry
single_axis_geometry = geometry_mrc_data(data_extent=data_extent,
                                         data_shape=data.shape,
                                         det_pix_size=det_pix_size,
                                         units='physical',
                                         extended_header=extended_header,
                                         downsam=downsam)

# --- Creating reconstruction space --- #

# Voxels in 3D region of interest
rec_shape = (data.shape[0], data.shape[0], data.shape[0])

## Create reconstruction extent
## for rod 
#min_pt = np.asarray((-150, -150, -150), float)
#max_pt = np.asarray((150, 150, 150), float)

# Create reconstruction extent
# for triangle, sphere
rec_extent = np.asarray(rec_shape, float)
#min_pt = np.asarray((-100, -100, -100), float)
#max_pt = np.asarray((100, 100, 100), float)

# Reconstruction space
rec_space = uniform_discr(-rec_extent / 2 * det_pix_size,
                          rec_extent / 2  * det_pix_size, rec_shape,
                          dtype='float32', interp='linear')

# --- Creating forward operator --- #

# Create forward operator
forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')

# --- Chaging the axises of the 3D data --- #

# Change the axises of the 3D data
#data = np.where(data >= 30, data, 0.0)
data_temp1 = np.swapaxes(data_downsam, 0, 2)
data_temp2 = np.swapaxes(data_temp1, 1, 2)
data_elem = forward_op.range.element(data_temp2)

# Show one sinograph
data_elem.show(title='Data in one projection',
               indices=np.s_[data_elem.shape[0] // 2, :, :])


#%%%
## --- Reconstructing by FBP --- #    
##
##
## Create FBP operator
#FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
#             frequency_scaling=1.0)
## Implement FBP method            
#rec_result_FBP = FBP(data_elem)
#rec_result_FBP_save = np.asarray(rec_result_FBP)
####rec_result_FBP_save = np.where(rec_result_FBP_save >= 0.1, rec_result_FBP_save, 0.0)
#### Shows result of FBP reconstruction
#rec_result_FBP.show(title='Filtered backprojection',
#                    indices=np.s_[:, :, rec_result_FBP.shape[-1] // 2],
#                    aspect='equal')
#rec_result_FBP.show(title='Filtered backprojection',
#                    indices=np.s_[:, rec_result_FBP.shape[1] // 2, :],
#                    aspect='equal')
#rec_result_FBP.show(title='Filtered backprojection',
#                    indices=np.s_[rec_result_FBP.shape[0] // 2 - 13, :, :],
#                    aspect='equal')
#
#### --- Save FBP reconstructed result --- #  
###  
##result_2_nii_format(result=rec_result_FBP_save,
##                    file_name='rod_FBPrecon_angle6.nii')
#result_2_mrc_format(result=rec_result_FBP_save,
#                    file_name='triangle_FBPrecon_angle6_8.mrc')


#%%%
# --- Reconstructing by LDDMM-based method --- #    

## Get the path of template
#directory2 = '/home/chchen/odl/odl/deform/'
#data_filename2 = 'triangle_LDDMMrecon_angle151_iter200.mrc'
#file_path2 = directory2 + data_filename2
#
## Create the template and show one slice
#data2, data_extent2, header2 = read_mrc_data(file_path=file_path2)
#template = rec_space.element(data2)

## sphere for rod, triangle, sphere2 for sphere
template = sphere(rec_space, smooth=True, taper=10.0)

## Create the ground truth and show one slice
#ground_truth = cube(rec_space, smooth=True, taper=20.0)
#ground_truth.show('ground truth, sphere smooth=True',
#                  indices=np.s_[rec_space.shape[0] // 2, :, :])

## Create data from ground truth
#data_template = forward_op(forward_op.domain.one())
## Show one sinograph
#data_template.show(title='Data in one projection',
#               indices=np.s_[data_elem.shape[0] // 2, :, :])

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl = 'geom'

if impl == 'mp':
    # Evaluate the mass from data
    mean = np.sum(data_temp2[data_temp2.shape[0] // 2]) / data_temp2[data_temp2.shape[0] // 2].size
    temp = data_temp2[data_temp2.shape[0] // 2] - mean
    temp = np.where(temp >= 0.01, data_temp2[data_temp2.shape[0] // 2], 0.0)
    mass_from_data = np.sum(temp)
    
    # Evaluate the mass of template
    mass_template = np.sum(np.asarray(template))
    
    # Get the same mass for template
    template = mass_from_data / mass_template * template

# Show slices of template
template.show('template, sphere smooth=True',
              indices=np.s_[rec_space.shape[0] // 2, :, :], aspect='equal')
template.show('template, sphere smooth=True',
              indices=np.s_[:, rec_space.shape[1] // 2, :], aspect='equal')
template.show('template, sphere smooth=True',
              indices=np.s_[:, :, rec_space.shape[-1] // 2], aspect='equal')

# Show intermiddle results
callback = (CallbackPrintIteration() &
            CallbackShow(indices=np.s_[:, :, rec_space.shape[-1] // 2], aspect='equal') &
            CallbackShow(indices=np.s_[:, rec_space.shape[1] // 2, :], aspect='equal') &
            CallbackShow(indices=np.s_[rec_space.shape[0] // 2, :, :], aspect='equal'))

# Maximum iteration number
niter = 20

# Give step size for solver
eps = 0.0005

# Give regularization parameter
lamb = 0.0000001

# Give the number of time points
time_itvs = 20

sigma = 0.5

# Give kernel function
def kernel(x):
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

rec_result = template

# Compute by LDDMM solver
image_N0, E = LDDMM_gradient_descent_solver(forward_op, data_elem, rec_result,
                                            time_itvs, niter, eps, lamb,
                                            kernel, impl, callback)

rec_result_1 = rec_space.element(image_N0[time_itvs // 3])
rec_result_2 = rec_space.element(image_N0[time_itvs * 2 // 3])

rec_result = rec_space.element(image_N0[time_itvs])
rec_result_save = np.asarray(rec_result)
#rec_result_save = np.where(rec_result_save >= 0.64, rec_result_save, 0.0)

rec_result.show('rec_result', indices=np.s_[:, :, rec_result.shape[-1] // 2], aspect='equal')
rec_result.show('rec_result', indices=np.s_[:, rec_result.shape[1] // 2, :], aspect='equal')
rec_result.show('rec_result', indices=np.s_[rec_result.shape[0] // 2, :, :], aspect='equal')

# --- Saving reconstructed result --- #  
      

#result_2_nii_format(result=rec_result_save,
#                    file_name='rod_LDDMMrecon_angle6_iter50.nii')
result_2_mrc_format(result=rec_result_save,
                    file_name='triangle_LDDMMrecon_angle151_iter200_kernel0_5_size20.mrc')


# --- Showing reconstructed result --- #  


# Plot the results of interest
plt.figure(1, figsize=(21, 21))
plt.clf()

plt.subplot(2, 2, 1)
plt.imshow(np.rot90(np.asarray(template)[template.shape[0] // 2, :, :]),
           cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max()) 
plt.colorbar()
plt.title('time_pts = {!r}'.format(0))

plt.subplot(2, 2, 2)
plt.imshow(np.rot90(np.asarray(rec_result_1)[rec_result_1.shape[0] // 2, :, :]),
           cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max()) 
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 3))

plt.subplot(2, 2, 3)
plt.imshow(np.rot90(np.asarray(rec_result_2)[rec_result_2.shape[0] // 2, :, :]),
           cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max()) 
plt.colorbar()
plt.title('time_pts = {!r}'.format(time_itvs // 3 * 2))

plt.subplot(2, 2, 4)
plt.imshow(np.rot90(np.asarray(rec_result)[rec_result.shape[0] // 2, :, :]),
           cmap='bone',
           vmin=np.asarray(template).min(),
           vmax=np.asarray(template).max()) 
plt.colorbar()
plt.title('Reconstructed image by {!r} iters, '
    '{!r} projs'.format(niter, single_axis_geometry.partition.shape[0]))

plt.figure(2, figsize=(8, 1.5))
plt.clf()
plt.plot(E)
plt.ylabel('Energy')
# plt.gca().axes.yaxis.set_ticklabels(['0']+['']*8)
plt.gca().axes.yaxis.set_ticklabels([])
plt.grid(True)

#
##%%%
### --- Reconstructing by TV method --- #    
##
##
### Initialize gradient operator
##grad_op = Gradient(rec_space, method='forward', pad_mode='symmetric')
##
### Column vector of two operators
##op = BroadcastOperator(forward_op, grad_op)
##
### Do not use the g functional, set it to zero.
##g = ZeroFunctional(op.domain)
##
### Set regularization parameter
##lamb = 10.0
##
### Isotropic TV-regularization i.e. the l1-norm
##l1_norm = lamb * L1Norm(grad_op.range)
##
### l2-squared data matching
##l2_norm = L2NormSquared(forward_op.range).translated(data_elem)
##
### --- Select solver parameters and solve using Chambolle-Pock --- #
### Estimate operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
##op_norm = 1.1 * power_method_opnorm(op)
##
##niter = 1000  # Number of iterations
##tau = 1.0 / op_norm  # Step size for the primal variable
##sigma = 1.0 / op_norm  # Step size for the dual variable
##gamma = 0.5
##
### Choose a starting point
##x = forward_op.domain.zero()
###x = rec_result_FBP
##
### Create functionals for the dual variable
### Combine functionals, order must correspond to the operator K
##f = SeparableSum(l2_norm, l1_norm)
##
### Optionally pass callback to the solver to display intermediate results
##callback = (CallbackPrintIteration() &
##            CallbackShow(indices=np.s_[:, :, x.shape[-1] // 2]) &
##            CallbackShow(indices=np.s_[:, x.shape[1] // 2, :]) &
##            CallbackShow(indices=np.s_[x.shape[0] // 2, :, :]))
##
### Run the algorithm
##chambolle_pock_solver(x, f, g, op, tau=tau, sigma=sigma, niter=niter,
##                      gamma=gamma, callback=callback)
##    
##
#### Show final result
###x.show(coords=[x.shape[0] // 2, None, None],
###       title='Reconstructed result by TV with {!r} iterations'.format(niter))
###x.show(coords=[None, x.shape[1] // 2, None],
###       title='Reconstructed result by TV with {!r} iterations'.format(niter))
###x.show(coords=[None, None, x.shape[-1] // 2],
###       title='Reconstructed result by TV with {!r} iterations'.format(niter))
##
##rec_result_save = np.asarray(x)
###rec_result_save = np.where(rec_result_save >= 0.64, rec_result_save, 0.0)
##
##x.show('rec_result', indices=np.s_[:, :, x.shape[-1] // 2])
##x.show('rec_result', indices=np.s_[:, x.shape[1] // 2, :])
##x.show('rec_result', indices=np.s_[x.shape[0] // 2, :, :])
##
##
### --- Saving reconstructed result --- #  
##      
##
###result_2_nii_format(result=rec_result_save, file_name='rod_TV_angle6.nii')
##result_2_mrc_format(result=rec_result_save, file_name='triangle_TV_angle6_6.mrc')
