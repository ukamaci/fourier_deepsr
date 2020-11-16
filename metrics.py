import numpy as np
import torch
import torch.fft
from torch import nn

def ringsum(im, corners=False):
    '''
    Given a 2d square array, calculate the sum of elements for each concentric
    ring of 1 pixel width and return the array of sums.

    Parameters
    ----------
    im : ndarray
        Elements to sum.
    corners : bool, optional
        If set to True, the largest ring diameter will be set to corner-to-corner
        distance; otherwise edge-to-edge.

    Returns
    -------
    ndarray
        1d array whose n^th element is the sum of input's elements inside the
        n^th ring.
    '''
    assert im.shape[0] == im.shape[1], 'input should be square'
    imsize = im.shape[0]
    r = np.arange(imsize) - imsize//2
    # generate a meshgrid where the origin is the midpoint of the array.
    # if the array length is even, the lower right point will be the origin.
    [xx,yy] = np.meshgrid(r,r)
    radii = np.sqrt(xx**2 + yy**2)

    maxrad = int(np.max(radii)) if corners else imsize//2
    sums = []

    for radius in range(maxrad+1):
        sums.append(
            np.sum(
                im[
                    np.where((radii < radius+0.5) & (radii >= radius - 0.5))
                ]
            )
        )

    return np.array(sums)

def get_frc(im1, im2, corners=False):
    '''
    Given two 2d arrays of the same shape, calculate the Fourier ring
    correlation between them.

    Parameters
    ----------
    im1 : ndarray
        The first array.
    im2 : ndarray
        The second array.
    corners : bool, optional
        If set to True, the largest ring diameter will be set to corner-to-corner
        distance; otherwise edge-to-edge.

    Returns
    -------
    ndarray
        1d complex array of normalized cross correlations between the rings of
        the input arrays.
    '''
    assert im1.shape == im2.shape, 'image shapes must match'
    assert im1.shape[0] == im1.shape[1], 'images must be square'

    im1f = np.fft.fftshift(np.fft.fft2(im1))
    im2f = np.fft.fftshift(np.fft.fft2(im2))

    return (
        ringsum(im1f * im2f.conj(), corners=corners) /
        np.sqrt(
            ringsum(abs(im1f)**2, corners=corners) *
            ringsum(abs(im2f)**2, corners=corners)
        )
    )

# Torch FFTshift implemetation
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real = x.real
    imag = x.imag
    for dim in range(1, len(real.size())):
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def ringsum_torch(im, corners=False):
    '''
    Given a 2d square array, calculate the sum of elements for each concentric
    ring of 1 pixel width and return the array of sums.

    Parameters
    ----------
    im : ndarray
        Elements to sum.
    corners : bool, optional
        If set to True, the largest ring diameter will be set to corner-to-corner
        distance; otherwise edge-to-edge.

    Returns
    -------
    ndarray
        1d array whose n^th element is the sum of input's elements inside the
        n^th ring.
    '''
    assert im.shape[0] == im.shape[1], 'input should be square'
    imsize = im.shape[0]
    r = torch.arange(imsize) - imsize//2
    # generate a meshgrid where the origin is the midpoint of the array.
    # if the array length is even, the lower right point will be the origin.
    [xx,yy] = torch.meshgrid(r,r)
    xx = xx.double()
    yy = yy.double()
    radii = torch.sqrt(xx**2 + yy**2)

    maxrad = int(torch.max(radii)) if corners else imsize//2
    sums = torch.zeros(maxrad+1)

    for radius in range(maxrad+1):
        sums[radius] = torch.sum(im[torch.where(torch.BoolTensor((radii < radius+0.5) & (radii >= radius - 0.5)))])

    return sums

def get_frc_torch(im1, im2, corners=False):
    '''
    Given two 2d arrays of the same shape, calculate the Fourier ring
    correlation between them.

    Parameters
    ----------
    im1 : ndarray
        The first array.
    im2 : ndarray
        The second array.
    corners : bool, optional
        If set to True, the largest ring diameter will be set to corner-to-corner
        distance; otherwise edge-to-edge.

    Returns
    -------
    ndarray
        1d complex array of normalized cross correlations between the rings of
        the input arrays.
    '''
    assert im1.shape == im2.shape, 'image shapes must match'
    assert im1.shape[0] == im1.shape[1], 'images must be square'

    im1f = batch_fftshift2d(torch.fft.fftn(im1))
    im2f = batch_fftshift2d(torch.fft.fftn(im2))

    return (
        ringsum_torch(im1f * im2f.conj(), corners=corners) /
        torch.sqrt(
            ringsum_torch(torch.abs(im1f)**2, corners=corners) *
            ringsum_torch(torch.abs(im2f)**2, corners=corners)
        )
    )

# def frc_loss(batch1, batch2):
#     loss = 0
#     for batch in range(batch1.shape[0]):
#         for ch in range(batch1.shape[1]):
#             im1 = batch1[batch,ch,:,:]
#             im2 = batch2[batch,ch, :, :]
#
#             loss += get_frc_torch(im1, im2)
#
#     return loss

# class frc_loss(nn.Module):
#     def __init__(self):
#         super(frc_loss, self).__init__()
#
#     def forward(self, batch1,batch2):
#         loss = 0
#         for batch in range(batch1.shape[0]):
#             for ch in range(batch1.shape[1]):
#                 im1 = batch1[batch, ch, :, :]
#                 im2 = batch2[batch, ch, :, :]
#
#                 loss += get_frc_torch(im1, im2)
#         return loss