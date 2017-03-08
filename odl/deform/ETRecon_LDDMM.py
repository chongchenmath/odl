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
from odl.phantom import sphere, sphere2, cube
from odl.solvers import CallbackShow, CallbackPrintIteration
from odl.deform.LDDMM_gradiant_descent_scheme import (
        LDDMM_gradient_descent_solver)
from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
from odl.tomo import RayTransform, fbp_op
standard_library.install_aliases()


# --- Reading data --- #

# Get the path of data
directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
data_filename = 'triangle.mrc'
file_path = directory + data_filename
data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
                                                           force_type='FEI1',
                                                           normalize=True)

#Downsample the data
downsam = 15
data_downsam = data[:, :, ::downsam]

# --- Getting geometry --- #

# Create 3-D parallel projection geometry
single_axis_geometry = geometry_mrc_data(data_extent=data_extent,
                                         data_shape=data.shape,
                                         extended_header=extended_header,
                                         downsam=downsam)

# --- Creating reconstruction space --- #

# Voxels in 3D region of interest
rec_shape = (128, 128, 128)

# Create reconstruction extent
rec_extent = np.asarray((1024, 1024, 1024), float)

# Reconstruction space
rec_space = uniform_discr(-rec_extent / 2, rec_extent / 2, rec_shape,
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


## --- Reconstructing by FBP --- #    
##
##
## Create FBP operator
#FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
#             frequency_scaling=1.0)
## Implement FBP method            
#rec_result_FBP = FBP(data_elem)
## Shows result of FBP reconstruction
#rec_result_FBP.show(title='Filtered backprojection',
#                    indices=np.s_[:, :, rec_result_FBP.shape[-1] // 2])
#rec_result_FBP.show(title='Filtered backprojection',
#                    indices=np.s_[:, rec_result_FBP.shape[1] // 2, :])
#rec_result_FBP.show(title='Filtered backprojection',
#                    indices=np.s_[rec_result_FBP.shape[0] // 2, :, :])
##
### --- Save FBP reconstructed result --- #  
##  
#result_2_nii_format(result=rec_result_FBP,
#                    file_name='triangle_FBPrecon.nii')
#result_2_mrc_format(result=rec_result_FBP, 
#                    file_name='triangle_FBPrecon.mrc')
#
#
## --- Reconstructing by LDDMM-based method --- #    
#
#
# Create the template and show one slice
template = sphere(rec_space, smooth=True, taper=50.0)
#template.show('template, sphere smooth=True',
#              indices=np.s_[rec_space.shape[0] // 2, :, :])


## Create the ground truth and show one slice
#ground_truth = cube(rec_space, smooth=True, taper=20.0)
#ground_truth.show('ground truth, sphere smooth=True',
#                  indices=np.s_[rec_space.shape[0] // 2, :, :])

## Create data from ground truth
#data_template = forward_op(forward_op.domain.one())
## Show one sinograph
#data_template.show(title='Data in one projection',
#               indices=np.s_[data_elem.shape[0] // 2, :, :])


# Maximum iteration number
niter = 50

# Implementation method for mass preserving or not,
# impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
# 'geom' means geometric deformation method
impl = 'geom'

# Show intermiddle results
callback = CallbackShow(
    '{!r} iterates'.format(impl), display_step=5) & CallbackPrintIteration()

# Give step size for solver
eps = 0.005

# Give regularization parameter
lamb = 0.0000001

# Give the number of time points
time_itvs = 20

# Give kernel function
def kernel(x):
    sigma = 5.0
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
rec_result_save = np.where(rec_result_save >= 0.5, rec_result_save, 0.0)

rec_result.show('rec_result', indices=np.s_[:, :, rec_result.shape[-1] // 2],clim=[0.001, 1.0])
rec_result.show('rec_result', indices=np.s_[:, rec_result.shape[1] // 2, :], clim=[0.001, 1.0])
rec_result.show('rec_result', indices=np.s_[rec_result.shape[0] // 2, :, :], clim=[0.001, 1.0])

# --- Saving reconstructed result --- #  
      

result_2_nii_format(result=rec_result_save, file_name='triangle_LDDMMrecon3.nii')
result_2_mrc_format(result=rec_result_save, file_name='triangle_LDDMMrecon3.mrc')


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
