from __future__ import print_function

import odl
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from odl.tomo.data import (FileReaderMRC, FileWriterMRC,
                           mrc_header_from_params)
from odl.tomo import Parallel3dAxisGeometry, RayTransform, fbp_op


# --- Reading --- #

# Get the path of data
directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
data_filename = 'rod.mrc'
file_path = directory + data_filename

with FileReaderMRC(file_path) as reader:
    header, data = reader.read()
    extended_header = reader.read_extended_header(force_type='FEI1')
    data_shape = reader.data_shape
    data_csides = reader.cell_sides_angstrom

# Get the extent of the data space
data_extent = data_csides * data_shape

# Create data space
data_space = odl.uniform_discr(-data_extent / 2, data_extent / 2, data_shape,
                               dtype='float32', interp='linear')

# Positive the values of the data
data = (data + 32768.0) / 32768.0

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
single_axis_geometry = Parallel3dAxisGeometry(angle_partition,
                                              detector_partition,
                                              axis=[0, 0, 1])

# Print the header and extended_header
#for key, value in header.items():
#        print(key, value)
#        
#for key, value in extended_header.items():
#        print(key, value)

# Reconstruction space
# Voxels in 3D region of interest
rec_shape = (362, 362, 362)

# Create reconstruction extent
rec_extent = np.asarray(rec_shape, float)

# Reconstruction space
rec_space = odl.uniform_discr(-rec_extent / 2, rec_extent / 2, rec_shape,
                              dtype='float32', interp='linear')

# Create forward operator
forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')

data_temp1 = np.swapaxes(data, 0, 2)
data_temp2 = np.swapaxes(data_temp1, 1, 2)
data_elem = forward_op.range.element(data_temp2)

# data_elem.show(indices=np.s_[:, :, data_shape[-1] // 2])
data_elem.show(indices=np.s_[10, :, :])


# --- Create FilteredBackProjection (FBP) operator --- #    
# Create FBP operator
FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
             frequency_scaling=1.0)

# Calculate filtered backprojection of data             
fbp_reconstruction = FBP(data_elem)

# Shows result of FBP reconstruction
fbp_reconstruction.show(title='Filtered backprojection', indices=np.s_[:, :, 36])


# --- Save reconstructed result --- #    
nib_arr = nib.Nifti1Image(np.asarray(fbp_reconstruction), affine=np.eye(4))
nib.save(nib_arr, 'rod_recon.nii')


# Write the stuff to a temporary (MRC) file
header = mrc_header_from_params(fbp_reconstruction.shape,
                                fbp_reconstruction.dtype, kind='volume')
out_file = open('rod_recon.mrc', 'wb+')

with FileWriterMRC(out_file, header) as writer:
    # Write both header and data to the file
    writer.write(fbp_reconstruction)

    # Check if the file size is consistent with header and data sizes
    print('File size ({}) = Header size ({}) + Data size ({})'
          ''.format(writer.file.seek(0, 2), writer.header_size,
                    fbp_reconstruction.nbytes))