from __future__ import print_function

import odl
import matplotlib.pyplot as plt
import numpy as np

from odl.tomo.data import (FileReaderMRC, FileWriterMRC,
                           mrc_header_from_params)
from odl.tomo import Parallel3dAxisGeometry, RayTransform, fbp_op

# --- Reading --- #
directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
#data_filename = 'rod.mrc'
data_filename = 'triangle.mrc'
file_path = directory + data_filename

# File readers can be used as context managers like `open`. As argument,
# either a file stream or a file name string can be used.

with FileReaderMRC(file_path) as reader:
    # Get header and data
    header, data = reader.read()
    extended_header = reader.read_extended_header(force_type='FEI1')

    # Print some interesting header information conveniently available
    # as reader attributes.
    print('Data shape: ', reader.data_shape)
    print('Data dtype: ', reader.data_dtype)
    print('Data axis ordering: ', reader.data_axis_order)
    print('Header size (bytes): ', reader.header_size)
    print('Additional text labels: ')
    print('')
    for label in reader.labels:
        if label.strip():
            print(repr(label))
    print('')

reader = FileReaderMRC(file_path)
header, tiltseries_data = reader.read()
extended_header = reader.read_extended_header(force_type='FEI1')

data_shape = reader.data_shape
data_csides = reader.cell_sides_angstrom
data_extent = data_csides * data_shape

# Create data space
data_space = odl.uniform_discr(-data_extent / 2, data_extent / 2, data_shape,
                               dtype='float32', interp='linear')
data_elem = data_space.element(data)

# data_elem.show(indices=np.s_[:, :, data_shape[-1] // 2])
data_elem.show(indices=np.s_[:, :, 150])

# Generate sampling on detector region, assume (0,0) is in the middle
detector_partition = odl.uniform_partition(-data_extent[0:2] / 2,
                                           data_extent[0:2] / 2,
                                           data_shape[0:2])  
# Single axis
# Have 151 angles uniformly distributed from -74.99730682 to 74.99730682
angle_partition = odl.uniform_partition(np.deg2rad(extended_header['a_tilt'][0]), 
                                        np.deg2rad(extended_header['a_tilt'][data_shape[-1]-1]),
                                        data_shape[-1],
                                        nodes_on_bdry=[(True, True)])

# Create 3-D parallel projection geometry
single_axis_geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition,
                                                       detector_partition,
                                                       axis=[0, 0, -1])

##plt.plot(extended_header['a_tilt'])
##plt.plot(extended_header['b_tilt'])
##plt.plot(extended_header['tilt_axis'])
#
## Reconstruction space
# Voxels in 3D region of interest
rec_shape = data_shape

# Create reconstruction extent
rec_extent = data_extent

# Reconstruction space
rec_space = odl.uniform_discr(-rec_extent / 2, rec_extent / 2, rec_shape,
                              dtype='float32', interp='linear')

## Create forward operator
forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')

# --- Create FilteredBackProjection (FBP) operator --- #    
# Create FBP operator
FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
             frequency_scaling=0.8)
# Calculate filtered backprojection of data             
fbp_reconstruction = FBP(data_elem)

# Shows result of FBP reconstruction
fbp_reconstruction.show(title='Filtered backprojection')

### Load data as an element in range of forward operator
#tiltseries = forward_op.range.element(np.swapaxes(tiltseries_data, 0, 2))
##tiltseries_nonoise = forward_op.range.element(
##    np.swapaxes(tiltseries_nonoise_data, 0, 2))
## Show 2D data for middle angle
#(tiltseries - np.mean(tiltseries)).show(
#    indices=np.s_[tiltseries.shape[0] // 2, :, :],
#    title='Middle projection (noise)')
##(tiltseries_nonoise - np.mean(tiltseries_nonoise)).show(
##    indices=np.s_[tiltseries_nonoise.shape[0] // 2, :, :],
##    title='Middle projection (noise-free)')
