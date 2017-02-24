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
from odl.discr import uniform_discr
from odl.tomo import RayTransform, fbp_op
from odl.phantom import sphere
from odl.solvers import CallbackShow, CallbackPrintIteration
from odl.deform.LDDMM_gradiant_descent_scheme import (
        LDDMM_gradient_descent_solver)
from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
standard_library.install_aliases()


# --- Reading -data-- #

# Get the path of data
directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
data_filename = 'triangle.mrc'
file_path = directory + data_filename
data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
                                                           force_type='FEI1',
                                                           normalize=True)

# --- Getting geometry --- #

# Create 3-D parallel projection geometry
single_axis_geometry = geometry_mrc_data(data_extent=data_extent,
                                         data_shape=data.shape,
                                         extended_header=extended_header)

# --- Creating reconstruction space --- #

# Voxels in 3D region of interest
rec_shape = (181, 181, 181)
# Create reconstruction extent
rec_extent = np.asarray(rec_shape, float)
# Reconstruction space
rec_space = uniform_discr(-rec_extent / 2, rec_extent / 2, rec_shape,
                          dtype='float32', interp='linear')

# --- Creating forward operator --- #

# Create forward operator
forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')

# --- Chaging the axises of the 3D data --- #

# Change the axises of the 3D data
data_temp1 = np.swapaxes(data, 0, 2)
data_temp2 = np.swapaxes(data_temp1, 1, 2)
data_elem = forward_op.range.element(data_temp2)
# Show one sinograph
data_elem.show(title='Data in one projection',
               indices=np.s_[data_elem.shape[0] // 2, :, :])


## --- Reconstructing by FBP --- #    
#
#
## Create FBP operator
#FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
#             frequency_scaling=1.0)
## Implement FBP method            
#rec_result = FBP(data_elem)
## Shows result of FBP reconstruction
#rec_result.show(title='Filtered backprojection',
#                        indices=np.s_[:, :, rec_result.shape[-1] // 2])
#
## --- Save FBP reconstructed result --- #  
#  
#result_2_nii_format(result=rec_result,
#                    file_name='triangle_FBPrecon.nii')
#result_2_mrc_format(result=rec_result, 
#                    file_name='triangle_FBPrecon.mrc')


# --- Reconstructing by LDDMM-based method --- #    


# Create the template and show one slice
template = sphere(rec_space, smooth=True)
template.show('sphere smooth=True', indices=np.s_[rec_space.shape[-1] // 2, :, :])

# Maximum iteration number
niter = 100

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl = 'geom'

# Show intermiddle results
callback = CallbackShow(
    '{!r} iterates'.format(impl), display_step=5) & CallbackPrintIteration()

# Give step size for solver
eps = 0.05

# Give regularization parameter
lamb = 0.0000001

# Create the gradient operator for the L2 functional
gradS = forward_op.adjoint * (forward_op - data_elem)

# Give the number of time points
time_itvs = 10

# Give kernel function
def kernel(x):
    sigma = 2.0
    scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
    return np.exp(-sum(scaled))

# Compute by LDDMM solver
image_N0 = LDDMM_gradient_descent_solver(
        gradS, template, time_itvs, niter, eps, lamb, kernel, impl, callback)

#rec_result_1 = rec_space.element(image_N0[time_itvs // 3])
#rec_result_2 = rec_space.element(image_N0[time_itvs * 2 // 3])
#rec_result = rec_space.element(image_N0[time_itvs])
#
#
## --- Saving reconstructed result --- #  
#      
#
#result_2_nii_format(result=rec_result, file_name='triangle_recon.nii')
#result_2_mrc_format(result=rec_result, file_name='triangle_recon.mrc')
#
#
## --- Showing reconstructed result --- #  
#
#
## Plot the results of interest
#plt.figure(1, figsize=(21, 21))
#plt.clf()
#
#plt.subplot(2, 2, 1)
#plt.imshow(np.rot90(template), cmap='bone',
#           vmin=np.asarray(template).min(),
#           vmax=np.asarray(template).max())
#plt.colorbar()
#plt.title('Template')
#
#plt.subplot(2, 2, 2)
#plt.imshow(np.rot90(rec_result_1), cmap='bone',
#           vmin=np.asarray(template).min(),
#           vmax=np.asarray(template).max()) 
#plt.colorbar()
#plt.title('time_pts = {!r}'.format(8))
#
#plt.subplot(2, 2, 3)
#plt.imshow(np.rot90(rec_result_2), cmap='bone',
#           vmin=np.asarray(template).min(),
#           vmax=np.asarray(template).max()) 
#plt.colorbar()
#plt.title('time_pts = {!r}'.format(15))
#
#plt.subplot(2, 2, 4)
#plt.imshow(np.rot90(rec_result), cmap='bone',
#           vmin=np.asarray(template).min(),
#           vmax=np.asarray(template).max()) 
#plt.colorbar()
#plt.title('Reconstructed image by {!r} iters, '
#    '{!r} projs'.format(niter, single_axis_geometry.partition.shape[0]))
