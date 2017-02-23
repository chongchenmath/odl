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
from odl.discr import (Gradient, Divergence, uniform_discr,
                       uniform_partition, DiscreteLp)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
from odl.tomo import Parallel2dGeometry, RayTransform, fbp_op
from odl.phantom import (white_noise, disc_phantom, submarine,
                         shepp_logan, geometric)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.solvers import CallbackShow, CallbackPrintIteration
from odl.deform.linearized import _linear_deform
from odl.deform.mass_preserving import geometric_deform, mass_presv_deform

from odl.deform.LDDMM_gradiant_descent_scheme import (
        LDDMM_gradient_descent_solver)
from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
standard_library.install_aliases()


# --- Reading --- #

# Get the path of data
directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
data_filename = 'rod.mrc'
file_path = directory + data_filename
data, data_extent, header, extended_header = read_mrc_data(file_path=file_path,
                                                           force_type='FEI1',
                                                           normalize=True)

# --- Getting geometry --- #

# Create 3-D parallel projection geometry
single_axis_geometry = geometry_mrc_data(data_extent=data_extent,
                                         data_shape=data.shape,
                                         extended_header=extended_header)

# Reconstruction space

# Voxels in 3D region of interest
rec_shape = (362, 362, 362)

# Create reconstruction extent
rec_extent = np.asarray(rec_shape, float)

# Reconstruction space
rec_space = uniform_discr(-rec_extent / 2, rec_extent / 2, rec_shape,
                          dtype='float32', interp='linear')

# Create forward operator
forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')

# Change the axises of the 3D data
data_temp1 = np.swapaxes(data, 0, 2)
data_temp2 = np.swapaxes(data_temp1, 1, 2)
data_elem = forward_op.range.element(data_temp2)

# Show a sinograph
data_elem.show(indices=np.s_[10, :, :])

# Create the template as the given image
template = rec_space.element(I1)

# --- Save reconstructed result --- #  
  
result_2_nii_format(result=fbp_reconstruction, file_name='rod_recon.nii')
result_2_mrc_format(result=fbp_reconstruction, file_name='rod_recon.mrc')
