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
Shape-based reconstruction using LDDMM.
"""

# Imports for common Python 2/3 codebase
from __future__ import print_function, division, absolute_import
from future import standard_library
import numpy as np
import matplotlib.pyplot as plt
from odl.discr import (Gradient, uniform_discr, ResizingOperator)
from odl.trafos import FourierTransform
from odl.space import ProductSpace
from odl.tomo import RayTransform
from odl.phantom import (white_noise, disc_phantom, submarine,
                         shepp_logan, geometric, sphere)
from odl.operator import (DiagonalOperator, IdentityOperator)
from odl.solvers import CallbackShow, CallbackPrintIteration
from odl.deform.linearized import _linear_deform
from odl.deform.mass_preserving import geometric_deform, mass_presv_deform
from odl.deform.mrc_data_io import (read_mrc_data, geometry_mrc_data,
                                    result_2_mrc_format, result_2_nii_format)
standard_library.install_aliases()


__all__ = ('LDDMM_gradient_descent_solver',)


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


def padded_ft_op(space, padded_size):
    """Create zero-padding Fourier transform.

    Parameters
    ----------
    space : the space needed to do Fourier transform.
    padded_size : the padded size in each axis.

    Returns
    -------
    padded_ft_op : `operator`
        Composed operator of Fourier transform composing with padded operator.
    """
    padded_op = ResizingOperator(
        space, ran_shp=[padded_size for _ in range(space.ndim)])
    shifts = [not s % 2 for s in space.shape]
    ft_op = FourierTransform(
        padded_op.range, halfcomplex=False, shift=shifts, impl='pyfftw')

    return ft_op * padded_op


def fitting_kernel(space, kernel):
    """Compute the vectorized discrete kernel ``K``.
    
    Parameters
    ----------
    space : the space used to define kernel ``K``.
    kernel : the used kernel function for data fitting term.
    
    Returns
    -------
    discretized_kernel : `ProductSpaceElement`
        The vectorized discrete kernel with the space dimension
    """
    kspace = ProductSpace(space, space.ndim)

    # Create the array of kernel values on the grid points
    discretized_kernel = kspace.element(
        [space.element(kernel) for _ in range(space.ndim)])
    return discretized_kernel


def LDDMM_gradient_descent_solver(forward_op, data_elem, template, time_pts,
                                  niter, eps, lamb, kernel, impl1='geom',
                                  impl2='least_square', callback=None):
    """
    Solver for the shape-based reconstruction using LDDMM.

    Notes
    -----
    The model is:
                
        .. math:: \min_{v} \lambda * \int_0^1 \|v(t)\|_V^2 dt + \|T(\phi_1.I) - g\|_2^2,

    where :math:`\phi_1.I := |D\phi_1^{-1}| I(\phi_1^{-1})` is for
    mass-preserving deformation, instead, :math:`\phi_1.I := I(\phi_1^{-1})`
    is for geometric deformation. :math:`\phi_1^{-1}` is the inverse of
    the solution at :math:`t=1` of flow of doffeomorphisms.
    :math:`|D\phi_1^{-1}|` is the Jacobian determinant of :math:`\phi_1^{-1}`.
    :math:`T` is the forward operator. If :math:`T` is an identity operator,
    the above model reduces to image matching. If :math:`T` is a non-identity
    forward operator, the above model is for shape-based image reconstrction.  
    :math:`g` is the detected data, `data_elem`. :math:`I` is the `template`.
    :math:`v(t)` is the velocity vector. :math:`V` is a reproducing kernel
    Hilbert space for velocity vector. :math:`lamb` is the regularization
    parameter. 
    

    Parameters
    ----------
    forward_op : `Operator`
        The forward operator of imaging.
    data_elem : `DiscreteLpElement`
        The given data.
    template : `DiscreteLpElement`
        Fixed template deformed by the deformation.
    time_pts : `int`
        The number of time intervals
    iter : `int`
        The given maximum iteration number.
    eps : `float`
        The given step size.
    lamb : `float`
        The given regularization parameter. It's a weighted value on the
        regularization-term side.
    kernel : `function`
        Kernel function in reproducing kernel Hilbert space.
    impl1 : {'geom', 'mp'}, optional
        The given implementation method for group action.
        The impl1 chooses 'mp' or 'geom', where 'mp' means using
        mass-preserving method, and 'geom' means using
        non-mass-preserving geometric method. Its defalt choice is 'geom'.
    impl2 : {'least square'}, optional
        The given implementation method for data matching term.
        Here the implementation only supports the case of least square.    
    callback : `class`, optional
        Show the intermediate results of iteration.

    Returns
    -------
    image_N0 : `ProductSpaceElement`
        The series of images produced by template and velocity field.
    mp_deformed_image_N0 : `ProductSpaceElement`
        The series of mass-preserving images produced by template
        and velocity field.
    E : `numpy.array`
        Storage of the energy values for iterations.
    """

    # Give the number of time intervals
    N = time_pts

    # Get the inverse of time intervals
    inv_N = 1.0 / N
    
    # Create the gradient operator for the squared L2 functional
    if impl2=='least_square':
        gradS = forward_op.adjoint * (forward_op - data_elem)
    else:
        raise NotImplementedError('now only support least square')

    # Create the space of images
    image_space = gradS.domain

    # Get the dimension of the space of images
    dim = image_space.ndim
    
    # Fourier transform setting for data matching term
    # The padded_size is the size of the padded domain 
    padded_size = 2 * image_space.shape[0]
    # The pad_ft_op is the operator of Fourier transform
    # composing with padded operator
    pad_ft_op = padded_ft_op(image_space, padded_size)
    # The vectorial_ft_op is a vectorial Fourier transform operator,
    # which constructs the diagnal element of a matrix.
    vectorial_ft_op = DiagonalOperator(*([pad_ft_op] * dim))
    
    # Compute the FT of kernel in fitting term
    discretized_kernel = fitting_kernel(image_space, kernel)
    ft_kernel_fitting = vectorial_ft_op(discretized_kernel)

    # Create the space for series deformations and series Jacobian determinant
    pspace = image_space.tangent_bundle
    series_pspace = ProductSpace(pspace, N+1)
    series_image_space = ProductSpace(image_space, N+1)

    # Initialize vector fileds at different time points
    vector_fields = series_pspace.zero()

    # Give the initial two series deformations and series Jacobian determinant
    image_N0 = series_image_space.element()
    grad_data_matching_N1 = series_image_space.element()
    grad_data_matching_const = image_space.element(gradS(template))

    if impl1=='geom':
        detDphi_N1 = series_image_space.element()
    elif impl1=='mp':
        detDphi_N0 = series_image_space.element()
        mp_deformed_image_N0 = series_image_space.element()
    else:
        raise NotImplementedError('unknown group action')

    for i in range(N+1):
        image_N0[i] = image_space.element(template)
        
        if impl1=='geom':
            detDphi_N1[i] = image_space.one()
        elif impl1=='mp':
            detDphi_N0[i] = image_space.one()
            mp_deformed_image_N0[i] = image_N0[i]

        grad_data_matching_N1[i] = grad_data_matching_const

    # Create the gradient operator
    grad_op = Gradient(domain=image_space, method='forward',
                       pad_mode='symmetric')

    # Create the divergence operator, which can be obtained from
    # the adjoint of gradient operator 
    # div_op = Divergence(domain=pspace, method='forward', pad_mode='symmetric')
    div_op = -grad_op.adjoint
    
    # Store energy
    E = []
    kE = len(E)
    E = np.hstack((E, np.zeros(niter)))

    # Begin iteration for non-mass-preserving case
    if impl1=='geom':
        print(impl1)
        for k in range(niter):
            # Update the velocity field
            for i in range(N+1):
                tmp1 = (grad_data_matching_N1[i] * detDphi_N1[i])
                tmp = grad_op(image_N0[i])

                for j in range(dim):
                    tmp[j] *= tmp1
                tmp3 = (2 * np.pi) ** (dim / 2.0) * vectorial_ft_op.inverse(
                    vectorial_ft_op(tmp) * ft_kernel_fitting)
    
                vector_fields[i] = (vector_fields[i] - eps * (
                    lamb * vector_fields[i] - tmp3))
    
            # Update image_N0 and detDphi_N1
            for i in range(N):
                # Update image_N0[i+1] by image_N0[i] and vector_fields[i+1]
                image_N0[i+1] = image_space.element(
                    _linear_deform(image_N0[i],
                                   -inv_N * vector_fields[i+1]))
#                # Update detDphi_N1[N-i-1] by detDphi_N1[N-i]
#                jacobian_det = image_domain.element(
#                    np.exp(inv_N * div_op(vector_fields[N-i-1])))
                jacobian_det = image_space.element(
                        1.0 + inv_N * div_op(vector_fields[N-i-1]))
                detDphi_N1[N-i-1] = (
                    jacobian_det * image_space.element(_linear_deform(
                        detDphi_N1[N-i], inv_N * vector_fields[N-i-1])))
            
            # Update the deformed template
            PhiStarI = image_N0[N]
    
            # Show intermediate result
            if callback is not None:
                callback(PhiStarI)

            # Compute the energy of the data fitting term 
            E[k+kE] += np.asarray((forward_op(PhiStarI) - data_elem)**2).sum()
    
            # Update gradient of the data matching: grad S(W_I(v^k))
            grad_data_matching_N1[N] = image_space.element(
                gradS(PhiStarI))
            for i in range(N):
                grad_data_matching_N1[N-i-1] = image_space.element(
                    _linear_deform(grad_data_matching_N1[N-i],
                                   inv_N * vector_fields[N-i-1]))
    
        return image_N0, E

    # Begin iteration for mass-preserving case
    elif impl1=='mp':
        print(impl1)
        for k in range(niter):
            # Update the velocity field
            for i in range(N+1):
                tmp = grad_op(grad_data_matching_N1[i])
                for j in range(dim):
                    tmp[j] *= mp_deformed_image_N0[i]
                tmp3 = (2 * np.pi) ** (dim / 2.0) * vectorial_ft_op.inverse(
                    vectorial_ft_op(tmp) * ft_kernel_fitting)
    
                vector_fields[i] = (vector_fields[i] - eps * (
                    lamb * vector_fields[i] + tmp3))

            # Update image_N0 and detDphi_N1
            for i in range(N):
                # Update image_N0[i+1] by image_N0[i] and vector_fields[i+1]
                image_N0[i+1] = image_space.element(
                    _linear_deform(image_N0[i], -inv_N * vector_fields[i+1])
                    )
#                # Update detDphi_N0[i+1] by detDphi_N0[i]
#                jacobian_det = image_domain.element(
#                    np.exp(-inv_N * div_op(vector_fields[i+1])))
                jacobian_det = image_space.element(
                        1.0 - inv_N * div_op(vector_fields[i+1]))
                detDphi_N0[i+1] = (jacobian_det * image_space.element(
                    _linear_deform(detDphi_N0[i],
                                   -inv_N * vector_fields[i+1])))
                mp_deformed_image_N0[i+1] = (image_N0[i+1] *
                    detDphi_N0[i+1])
            
            # Update the deformed template
            PhiStarI = mp_deformed_image_N0[N]
    
            # Show intermediate result
            if callback is not None:
                callback(PhiStarI)
            
            # Compute the energy of the data fitting term 
            E[k+kE] += np.asarray((forward_op(PhiStarI) - data_elem)**2).sum()
    
            # Update gradient of the data matching: grad S(W_I(v^k))
            grad_data_matching_N1[N] = image_space.element(
                gradS(PhiStarI))
            for i in range(N):
                grad_data_matching_N1[N-i-1] = image_space.element(
                    _linear_deform(grad_data_matching_N1[N-i],
                                   inv_N * vector_fields[N-i-1]))
    
        return mp_deformed_image_N0, E


if __name__ == '__main__':

    import odl
    # --- Reading data --- #

    # Get the path of data
    directory = '/home/chchen/SwedenWork_Chong/Data_S/wetransfer-569840/'
    data_filename = 'triangle.mrc'
    file_path = directory + data_filename
    data, data_extent, header, extended_header = odl.deform.mrc_data_io.read_mrc_data(
            file_path=file_path, force_type='FEI1', normalize=True)
    
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
    data_temp1 = np.swapaxes(data_downsam, 0, 2)
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
    template = sphere(rec_space, smooth=True, taper=10.0)
    template.show('sphere smooth=True', indices=np.s_[rec_space.shape[0] // 2, :, :])
    
    # Maximum iteration number
    niter = 50
    
    # Implementation method for mass preserving or not,
    # impl chooses 'mp' or 'geom', 'mp' means mass-preserving deformation method,
    # 'geom' means geometric deformation method
    impl1 = 'geom'
    
    impl2 = 'least_square'
    
    # Show intermiddle results
    callback = (CallbackPrintIteration() &
            CallbackShow(indices=np.s_[:, :, rec_space.shape[-1] // 2], aspect='equal') &
            CallbackShow(indices=np.s_[:, rec_space.shape[1] // 2, :], aspect='equal') &
            CallbackShow(indices=np.s_[rec_space.shape[0] // 2, :, :], aspect='equal'))

    
    # Give step size for solver
    eps = 0.001
    
    # Give regularization parameter
    lamb = 0.0000001
    
    # Give the number of time points
    time_itvs = 20
    
    # Give kernel function
    def kernel(x):
        sigma = 4.0
        scaled = [xi ** 2 / (2 * sigma ** 2) for xi in x]
        return np.exp(-sum(scaled))
    
    # Compute by LDDMM solver
    image_N0, E = LDDMM_gradient_descent_solver(forward_op, data_elem, template,
                                                time_itvs, niter, eps, lamb,
                                                kernel, impl1, impl2,  callback)
    
    rec_result_1 = rec_space.element(image_N0[time_itvs // 3])
    rec_result_2 = rec_space.element(image_N0[time_itvs * 2 // 3])
    rec_result = rec_space.element(image_N0[time_itvs])
    
    
    # --- Saving reconstructed result --- #  
          
    
    result_2_nii_format(result=rec_result, file_name='triangle_LDDMMrecon3.nii')
    result_2_mrc_format(result=rec_result, file_name='triangle_LDDMMrecon3.mrc')
    
    
    # --- Showing reconstructed result --- #  
    
    
    # Plot the results of interest
    plt.figure(1, figsize=(21, 21))
    plt.clf()
    
    plt.subplot(2, 2, 1)
    plt.imshow(np.asarray(template)[template.shape[0] // 2, :, :], cmap='bone',
               vmin=np.asarray(template).min(),
               vmax=np.asarray(template).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(0))
    
    plt.subplot(2, 2, 2)
    plt.imshow(np.asarray(rec_result_1)[rec_result_1.shape[0] // 2, :, :],
               cmap='bone',
               vmin=np.asarray(template).min(),
               vmax=np.asarray(template).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs // 3))
    
    plt.subplot(2, 2, 3)
    plt.imshow(np.asarray(rec_result_2)[rec_result_2.shape[0] // 2, :, :],
               cmap='bone',
               vmin=np.asarray(template).min(),
               vmax=np.asarray(template).max()) 
    plt.colorbar()
    plt.title('time_pts = {!r}'.format(time_itvs // 3 * 2))
    
    plt.subplot(2, 2, 4)
    plt.imshow(np.asarray(rec_result)[rec_result.shape[0] // 2, :, :],
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
