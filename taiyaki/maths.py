import numpy as np


def med_mad(data, factor=None, axis=None, keepdims=False):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median

    :param data: A :class:`ndarray` object
    :param factor: Factor to scale MAD by. Default (None) is to be consistent
    with the standard deviation of a normal distribution
    (i.e. mad( N(0,\sigma^2) ) = \sigma).
    :param axis: For multidimensional arrays, which axis to calculate over
    :param keepdims: If True, axis is kept as dimension of length 1

    :returns: a tuple containing the median and MAD of the data
    """
    if factor is None:
        factor = 1.4826
    dmed = np.median(data, axis=axis, keepdims=True)
    dmad = factor * np.median(abs(data - dmed), axis=axis, keepdims=True)
    if axis is None:
        dmed = dmed.flatten()[0]
        dmad = dmad.flatten()[0]
    elif not keepdims:
        dmed = dmed.squeeze(axis)
        dmad = dmad.squeeze(axis)
    return dmed, dmad


def mad(data, factor=None, axis=None, keepdims=False):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and (by default)
    adjust by a factor for asymptotically normal consistency.

    :param data: A :class:`ndarray` object
    :param factor: Factor to scale MAD by. Default (None) is to be consistent
    with the standard deviation of a normal distribution
    (i.e. mad( N(0,\sigma^2) ) = \sigma).
    :param axis: For multidimensional arrays, which axis to calculate the median over.
    :param keepdims: If True, axis is kept as dimension of length 1

    :returns: the (scaled) MAD
    """
    _ , dmad = med_mad(data, factor=factor, axis=axis, keepdims=keepdims)
    return dmad


def studentise(x, axis=None):
    """  Studentise a numpy array along a given axis
    :param x: A :class:`ndaray`
    :param axis: axis over which to studentise

    :returns: A :class:`nd.array` with same shape as x
    """
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    s = np.where(s > 0.0, s, 1.0)
    return np.divide(x - m, s)


def geometric_prior(n, m, rev=False):
    """ Log probabilities for random start time with geoemetric distribution

    :param n: length of output vector
    :param m: mean of distribution
    :param rev: reverse distribution

    :returns: A 1D :class:`ndarray` containing log probabilities
    """
    p = 1.0 / (1.0 + m)
    prior = np.repeat(np.log(p), n)
    prior[1:] += np.arange(1, n) * np.log1p(-p)
    if rev:
        prior = prior[::-1]
    return prior


def logsumexp(x, axis=None, keepdims=False):
    """ Calculate log-sum-exp of an array in a stable manner

    log-sum-exp = log( sum_i exp x_i )

    Calculation is stablised against under- and over-flow in the exponential by
    finding the maximum value of the array x_M and calculating:

    log-sump-exp = x_M + log( sum_i exp(x_i - x_M) )

    :param x: Array containing numbers whose log-sum-exp is desired
    :param axis: Axis or axes along which the log-sum-exp are computed. The default
                 is to compute the log-sum-exp of the flattened array.
    :param keepdims: If this is set to True, the axes which are reduced are left
                     in the result as dimensions with size one

    :returns: Array containing log-sum-exp
    """
    maxX = np.amax(x, axis=axis, keepdims=True)
    rem = np.log(np.sum(np.exp(x - maxX), axis=axis, keepdims=keepdims))
    maxX_out = maxX.reshape(np.shape(rem))
    return maxX_out + rem


def rle(x, tol=0):
    """  Run length encoding of array x

    Note: where matching is done with some tolerance, the first element
    of the run is chosen as representative.

    :param x: array
    :param tol: tolerance of match (for continuous arrays)

    :returns: tuple of array containing elements of x and array containing
    length of run
    """

    delta_x = np.ediff1d(x, to_begin=1)
    starts = np.where(np.absolute(delta_x) > tol)[0]
    last_runlength = len(x) - starts[-1]
    runlength = np.ediff1d(starts, to_end=last_runlength)

    return (x[starts], runlength)
