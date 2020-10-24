import numpy as np

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
