from __future__ import print_function

import odl
import matplotlib.pyplot as plt
import numpy as np

from odl.tomo.data import (FileReaderMRC, FileWriterMRC,
                           mrc_header_from_params)

# --- Reading --- #
directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
data_filename = 'rod.mrc'
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
data_space = odl.uniform_discr(-data_extent / 2, data_extent / 2, data_shape)
data_elem = data_space.element(data)

# data_elem.show(indices=np.s_[:, :, data_shape[-1] // 2])
data_elem.show(indices=np.s_[:, :, 75])

# Generate sampling on detector region, assume (0,0) is in the middle
detector_partition = odl.uniform_partition(-data_csides[0:2] / 2,
                                           data_csides[0:2] / 2,
                                           data_shape[0:2])  
# Have 151 angles uniformly distributed from -74.99730682 to 74.99730682
angle_partition = odl.uniform_partition(np.deg2rad(-74.99730682), 
                                        np.deg2rad(74.99730682), 151,
                                        nodes_on_bdry=[(True, True)])

# Create 3-D parallel projection geometry
single_axis_geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition,
                                                       detector_partition,
                                                       axis=[0, 0, -1])

#plt.plot(extended_header['a_tilt'])
#plt.plot(extended_header['b_tilt'])
#plt.plot(extended_header['tilt_axis'])

## Reconstruction space
# Voxels in 3D region of interest
rec_shape = np.array([200, 200, 200])
# Define the 3D region centered at the origin in nm
rec_size = np.array([70.,70., 70.])
# Reconstruction space
rec_space = odl.uniform_discr(-rec_size / 2, rec_size / 2, rec_shape,
                              dtype='float32', interp='linear')
                              
## Create forward operator
forward_op = odl.tomo.RayTransform(rec_space, single_axis_geometry,
                                   impl='astra_cuda')

## Load data as an element in range of forward operator
tiltseries = forward_op.range.element(np.swapaxes(tiltseries_data, 0, 2))
#tiltseries_nonoise = forward_op.range.element(
#    np.swapaxes(tiltseries_nonoise_data, 0, 2))
# Show 2D data for middle angle
(tiltseries - np.mean(tiltseries)).show(
    indices=np.s_[tiltseries.shape[0] // 2, :, :],
    title='Middle projection (noise)')
#(tiltseries_nonoise - np.mean(tiltseries_nonoise)).show(
#    indices=np.s_[tiltseries_nonoise.shape[0] // 2, :, :],
#    title='Middle projection (noise-free)')
