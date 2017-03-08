from __future__ import print_function
from future import standard_library
import numpy as np
from math import ceil
import nibabel as nib
from odl.tomo.data import (FileReaderMRC, FileWriterMRC,
                           mrc_header_from_params)
from odl.tomo import Parallel3dAxisGeometry, RayTransform, fbp_op
from odl.discr import uniform_discr, uniform_partition
standard_library.install_aliases()


__all__ = ('read_mrc_data', 'geometry_mrc_data',
           'result_2_mrc_format', 'result_2_nii_format',)


def read_mrc_data(file_path=None, force_type=None, normalize=None):
    """Read FEI MRC data from given file path.

    Parameters
    ----------
    file_path : string
        Given file path.
    force_type : string
        Support types of data, now `FEI1` or general mrc data.
    normalize : bool
        Whether normalize the data into [0, 1]. 

    Returns
    -------
    data : `numpy.ndarray`
        Projection data. Normalized into [0, 1].
    data_extent : `numpy.ndarray`
        Extent of the data. 
    header : OrderedDict
        MRC data head information.
    extended_header : OrderedDict
        FEI MRC data extented head information.
    """

    reader = FileReaderMRC(file_path)
    header, data = reader.read()

    data_shape = reader.data_shape
    data_csides = reader.cell_sides_angstrom
    data_extent = data_csides * data_shape
    
    if normalize == True:
        data = (data + 32768.0) / 327.68
    
    if force_type == 'FEI1':    
        extended_header = reader.read_extended_header(force_type='FEI1')
        return data, data_extent, header, extended_header
    else:
        return data, data_extent, header


def geometry_mrc_data(data_extent=None, data_shape=None,
                      extended_header=None, downsam=None):
    """Get the geometry of FEI MRC data.
    
    Now only support single axis tilt geometry. 

    Parameters
    ----------
    data_shape : int or sequence of ints
        Number of samples per axis.
    data_extent : float or sequence of floats
        The desired function domain for data.
    extended_header : OrderedDict
        FEI MRC data extented head information.

    Returns
    -------
    single_axis_geometry : class
        The scanning geometry of the data.
    """
    
    # Generate sampling on detector region, assume (0,0) is in the middle
    detector_partition = uniform_partition(-data_extent[0:2] / 2,
                                           data_extent[0:2] / 2,
                                           data_shape[0:2])  

    # Have 151 angles uniformly distributed from -74.99730682 to 74.99730682
    angle_partition = uniform_partition(np.deg2rad(extended_header['a_tilt'][0]), 
                                        np.deg2rad(extended_header['a_tilt'][data_shape[-1]-1]),
                                        ceil(data_shape[-1] / float(downsam)), 
                                        nodes_on_bdry=[(True, True)])

    # Create 3-D parallel projection geometry
    return Parallel3dAxisGeometry(angle_partition, detector_partition,
                                  axis=[0, 0, 1])


# Write the reconstructed result to an MRC file
def result_2_mrc_format(result=None, file_name=None):
    """Write reconstructed result into MRC format, such as '*.mrc'.

    Parameters
    ----------
    result : `numpy.ndarray` or element of space of `uniform_discr`
        A 3D volume result.
    file_name : string
        The file name.
    extended_header : OrderedDict
        FEI MRC data extented head information.
    """
    
    header = mrc_header_from_params(result.shape, result.dtype, kind='volume')
    out_file = open(file_name, 'wb+')
    writer = FileWriterMRC(out_file, header)
    # Write both header and data to the file
    writer.write(result)

#    # Check if the file size is consistent with header and data sizes
#    print('File size ({}) = Header size ({}) + Data size ({})'
#          ''.format(writer.file.seek(0, 2), writer.header_size,
#                    fbp_reconstruction.nbytes))


# Write the reconstructed result to a nii file
def result_2_nii_format(result=None, file_name=None):
    """Write reconstructed result into nii format, such as '*.nii'.
    
    Viewed by ITK-snap.

    Parameters
    ----------
    result : `numpy.ndarray` or element of space of `uniform_discr`
        A 3D volume result.
    file_name : string
        The file name.
    """
    nib_arr = nib.Nifti1Image(np.asarray(result), affine=np.eye(4))
    nib.save(nib_arr, file_name)


if __name__ == '__main__':

    # --- Reading --- #
    
    # Get the path of data
    directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
    data_filename = 'rod.mrc'
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
    
    # Reconstruction space
    
    # Voxels in 3D region of interest
    rec_shape = (128, 128, 128)
    
    # Create reconstruction extent
    rec_extent = np.asarray((1024, 1024, 1024), float)
    
    # Reconstruction space
    rec_space = uniform_discr(-rec_extent / 2, rec_extent / 2, rec_shape,
                              dtype='float32', interp='linear')
    
    # Create forward operator
    forward_op = RayTransform(rec_space, single_axis_geometry, impl='astra_cuda')
    
    # Change the axises of the 3D data
    data_temp1 = np.swapaxes(data_downsam, 0, 2)
    data_temp2 = np.swapaxes(data_temp1, 1, 2)
    data_elem = forward_op.range.element(data_temp2)
    
    # Show one sinograph
    data_elem.show(title='Data in one projection',
                   indices=np.s_[data_elem.shape[0] // 2, :, :])
    
    
    # --- Create FilteredBackProjection (FBP) operator --- #    
    
    # Create FBP operator
    FBP = fbp_op(forward_op, padding=True, filter_type='Hamming',
                 frequency_scaling=1.0)
    
    # Implement FBP method            
    fbp_reconstruction = FBP(data_elem)
    
    # Shows result of FBP reconstruction
    fbp_reconstruction.show(title='Filtered backprojection',
                            indices=np.s_[:, :, fbp_reconstruction.shape[-1] // 2])
    
    # --- Save reconstructed result --- #  
      
    result_2_nii_format(result=fbp_reconstruction, file_name='rod_recon.nii')
    result_2_mrc_format(result=fbp_reconstruction, file_name='rod_recon.mrc')
    
    # Run also the doctests
    # pylint: disable=wrong-import-position
    from odl.util.testutils import run_doctests
    run_doctests()
