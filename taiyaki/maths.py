import numpy as np
from collections import deque


MAD_SD_FACTOR = 1.4826


def med_mad(data, factor=None, axis=None, keepdims=False):
    """Compute the Median Absolute Deviation, i.e., the median
    of the absolute deviations from the median, and the median

    Args:
        data (:class:`ndarray`): Data from which to determine med/mad
        factor (float): Factor to scale MAD by. Default (None) is to be
            consistent with the standard deviation of a normal distribution
            (i.e. mad( N(0,sigma^2) ) = sigma).
        axis (int): For multidimensional arrays, which axis to calculate over
        keepdims (bool): If True, axis is kept as dimension of length 1

    :returns: a tuple containing the median and MAD of the data
    """
    if factor is None:
        factor = MAD_SD_FACTOR
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

    Args:
        data (:class:`ndarray`): Data from which to determine mad
        factor (float): Factor to scale MAD by. Default (None) is to be
            consistent with the standard deviation of a normal distribution
            (i.e. mad( N(0,sigma^2) ) = sigma).
        axis (int): For multidimensional arrays, which axis to calculate over
        keepdims (bool): If True, axis is kept as dimension of length 1

    Returns:
        Median Absolute Deviation (float; scaled if factor is set)
    """
    _, dmad = med_mad(data, factor=factor, axis=axis, keepdims=keepdims)
    return dmad


def logsumexp(x, axis=None, keepdims=False):
    """ Calculate log-sum-exp of an array in a stable manner

    log-sum-exp = log( sum_i exp x_i )

    Calculation is stablised against under- and over-flow in the exponential by
    finding the maximum value of the array x_M and calculating:

    log-sump-exp = x_M + log( sum_i exp(x_i - x_M) )

    Args:
        x: Array containing numbers whose log-sum-exp is desired
        axis: Axis or axes along which the log-sum-exp are computed. The
            default is to compute the log-sum-exp of the flattened array.
        keepdims: If this is set to True, the axes which are reduced are
            left in the result as dimensions with size one

    Returns:
        Array containing log-sum-exp
    """
    maxX = np.amax(x, axis=axis, keepdims=True)
    rem = np.log(np.sum(np.exp(x - maxX), axis=axis, keepdims=keepdims))
    maxX_out = maxX.reshape(np.shape(rem))
    return maxX_out + rem


def rle(x, tol=0):
    """  Run length encoding of array x

    Note: where matching is done with some tolerance, the first element
    of the run is chosen as representative.

    Args:
        x: array
        tol: tolerance of match (for continuous arrays)

    Returns:
        tuple of array containing elements of x and array containing
            length of run
    """

    delta_x = np.ediff1d(x, to_begin=1)
    starts = np.where(np.absolute(delta_x) > tol)[0]
    last_runlength = len(x) - starts[-1]
    runlength = np.ediff1d(starts, to_end=last_runlength)

    return (x[starts], runlength)


class RollingQuantile:
    """Calculate rolling quantile of time series over a specified window"""

    def __init__(
            self, upper_quantile, window=100, min_data=1, default_to=None):
        """Set up rolling quantile calculator. With the default settings, there
        is no minimum data length and the first call to the calculator
        returns the first value x, and the second call returns
         upper_quantile * a + (1-upper_quantile) * b
        where b>=a and (a,b) are the first two x values.

        Args:
            upper_quantile : if upper_quantile = 0.05 then we calculate the
                value exceeded by 5% of the data
            window : calculation is done over the last <window> data points
            min_data : if fewer than <min_data> data points available...
            default_to : ...then return this value.
        """
        self.window_data = deque()
        self.upper_quantile = upper_quantile
        self.window = window
        self.min_data = min_data
        self.default_returnvalue = default_to

    def update(self, x):
        """Update with time series value x and return rolling quantile."""
        self.window_data.append(x)
        if len(self.window_data) > self.window:
            self.window_data.popleft()
        if len(self.window_data) < self.min_data:
            return self.default_returnvalue
        return np.quantile(self.window_data, 1.0 - self.upper_quantile)


class RollingMAD:
    """ Calculate rolling meadian absolute deviation cap over a specified
    window for a vector of values

    For example compute gradient maxima by:

        parameters = [p for p in network.parameters() if p.requires_grad]
        rolling_mads = RollingMAD(len(parameters))
        grad_maxs = [
            float(torch.max(torch.abs(layer_params.grad.detach())))
            for layer_params in parameters]
        grad_max_threshs = rolling_mads.update(grad_maxs)
    """

    def __init__(self, nparams, n_mads=0, window=1000, default_to=None):
        """ Set up rolling MAD calculator.

        Args:
            nparams : Number of parameter arrays to track independently
            n_mads : Number of MADs above the median to return
            window : calculation is done over the last <window> data points
            default_to : Return this value before window values have been added
        """
        self.n_mads = n_mads
        self.default_to = default_to
        self._window_data = np.empty((nparams, window), dtype='f4')
        self._curr_iter = 0

    @property
    def nparams(self):
        return self._window_data.shape[0]

    @property
    def window(self):
        return self._window_data.shape[1]

    def update(self, vals):
        """ Update with time series values and return MAD thresholds.

        Returns:
            List of `median + (nmods * mad)` of the current window of data.
                Before window number of values have been added via update the
                `default_to` value is returned.
        """
        assert len(vals) == self.nparams, (
            'Number of values ({}) provided does not match number of ' +
            'parameters ({}).').format(len(vals), self.nparams)

        self._window_data[:, self._curr_iter % self.window] = vals
        self._curr_iter += 1
        if self._curr_iter < self.window:
            return self.default_to

        med, mad = med_mad(self._window_data, axis=1)
        return med + (mad * self.n_mads)
