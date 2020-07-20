from collections import OrderedDict
import numpy as np

from scipy import linalg
from scipy.stats import truncnorm
import torch
from torch import nn
from torch.nn import Parameter

from taiyaki import activation, flipflopfings
from taiyaki.config import taiyaki_dtype
from taiyaki.constants import LARGE_LOG_VAL


"""  Convention: inMat row major (C ordering) as (time, batch, state)
"""
_FORGET_BIAS = 2.0
#  Increment whenever layers change in non-compatible way.
#  Remember to alter misc/upgrade_model.py to enable upgrade of old models
MODEL_VERSION = 2


def get_function_name(fun):
    """  Get function name

    Args:
        fun (<function>): function for which name is required

    Returns:
        str: Name of function
    """
    if isinstance(fun, torch._C.Function):
        #  Function is JIT'd by pytorch
        return fun.name

    return fun.__name__


def init_(param, value):
    """  Set parameter value (inplace)

    Args:
        param (:torch:`Parameter`): Param to set inplace
        value (tensor, numpy array, list or tuple): Value to set `param` to

    Returns:
        None:  `param` set inplace
    """
    value_as_tensor = torch.tensor(value, dtype=param.data.dtype)
    with torch.no_grad():
        param.set_(value_as_tensor)


def random_orthonormal(n, m=None):
    """  Generate random orthonormal matrix

    Implementation in scipy.stats.ortho_group is rather slow. We use
    QR decomposition on a matrix of unit-variance Gaussian noise,
    followed by a sign-flipping trick which is a version of the
    normalisation recommended by `Mezzadri`_ for unitary matrices.

    .. _Mezzadri:
        https://arxiv.org/pdf/math-ph/0609050v2.pdf.

    A square matrix is generated if only one parameter is given, otherwise a
    rectangular matrix is generated.  The number of columns must be greater than
    the number of rows.

    Args:
        n (int): rank of matrix to generate
        m (int, optional): second dimension of matrix, set equal to `n` if None.

    Returns:
        `ndarray`: orthonormal matrix
    """
    m = n if m is None else m
    assert m >= n
    x = np.random.rand(m, m)
    Q, r = linalg.qr(x, mode='economic')
    # Make diag matrix which flips first element of each row of r to positive
    flipper = np.diag(np.sign(np.diag(r)))
    # Make square orthog matrix
    square_orthog = Q.dot(flipper)
    # Drop unneeded rows
    return square_orthog[:n, :]


def orthonormal_matrix(nrow, ncol):
    """  Generate random orthonormal matrix

    Note:
        Retangular matrices where the number of rows exceeds the number of
    columns are partitioned into square chunks as far as possible and each
    chunk initialised as a separate orthonormal matrix.  Any remaining columns
    are then initialised as a retangualr orthogonal matrix.

    Args:
        nrow (int): number of rows of matrix to generate.
        ncol (int): number of columns of matrix to generate.

    Returns:
        `ndarray`: orthonormal matrix
    """
    nrep = nrow // ncol
    out = np.zeros((nrow, ncol), dtype='f4')
    for i in range(nrep):
        out[i * ncol: i * ncol + ncol] = random_orthonormal(ncol)
    #  Remaining
    remsize = nrow - nrep * ncol
    if remsize > 0:
        out[nrep * ncol:, :] = random_orthonormal(remsize, ncol)

    return out


def truncated_normal(size, sd):
    """  Truncated normal for Xavier style initiation

    Generates random observations from a normal distribution truncated at
    +/- 2, scaled so that the standard deviation is `sd`.  The expectation
    of the distribution is zero.

    Args:
        size (int):  Number of observations to generate
        sd (float):  Standard deviation of of distribution

    Returns:
        `ndarray`: random observations from truncated normal distribution
    """
    res = sd * truncnorm.rvs(-2, 2, size=size)
    return res.astype('f4')


class Reverse(nn.Module):
    """  Reverse layer in reverse time

    Input and output of enclosed layer are flipped along the time axis.

    Attributes:
        layer (:nn:Module): Taiyaki layer to reverse.
    """

    def __init__(self, layer):
        """  Constructor for `Reverse` layer

        Args:
            layer (:nn:Module): Taiyaki layer to reverse
        """
        super().__init__()
        self.layer = layer

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        return torch.flip(self.layer(torch.flip(x, (0,))), (0,))

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', "reverse"),
                            ('sublayers', self.layer.json())])


class Residual(nn.Module):
    """  Apply a residual connection around a layer

        The enclosed layer is called and its input is added to its output to
    form a residual connection.  The insize and outsize of the enclosed layer
    must be the same.

    Attributes:
        layer (:nn:Module): Taiyaki layer to wrap.
    """

    def __init__(self, layer):
        """  Constructor for `Residual` layer

        Args:
            layer (:nn:Module): Taiyaki layer to wrap.  The size of the output
                of the layer must be equal to its input.
        """
        super().__init__()
        self.layer = layer

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        return x + self.layer(x)

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', "Residual"),
                            ('sublayers', self.layer.json())])


class GatedResidual(nn.Module):
    """  Add a tuneable residual connection around a layer

        sigmoid(alpha) * X + (1 - sigmoid(alpha)) * layer(X)

    Attributes:
        alpha (:nn:`Parameter`, scalar): Gating parameter to apply.
        layer (:nn:Module): Taiyaki layer to wrap.  The size of the output
            of the layer must be equal to its input.
    """

    def __init__(self, layer, gate_init=0.0):
        """  Constructor for `GatedResidual` layer

        Args:
            layer (:nn:Module): Taiyaki layer to wrap.  The size of the output
                of the layer must be equal to its input.
            gate_init (float, optional):  Initial value for gating parameter,
                default 0.0 is equal mix of input and output similar to
                "layer:`Residual`.
        """
        super().__init__()
        self.layer = layer
        self.alpha = Parameter(torch.tensor([gate_init]))

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        gate = activation.sigmoid(self.alpha)
        y = self.layer(x)
        return gate * x + (1 - gate) * y

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "GatedResidual"),
                           ('sublayers', self.layer.json())])
        res['params'] = OrderedDict(
            [('alpha', float(self.alpha.detach_().numpy()[0]))])
        return res


class FeedForward(nn.Module):
    """  Basic feedforward layer with activation and bias

         out = fun( inMat W + b )

    Attributes:
        insize (int):  Size (number of neurons) expected in layer input.
        size (int):  Size (number of neurons) of layer output.
        has_bias (bool, optional): Whether layer has bias.  If `False`, bias
            is initialised to zero and not trained.
        linear (:nn:`Module`):  Pytorch module implementing linear
            transform.
        activation (<function>, optional):  Activation function to apply to
            output linear transform.
    """

    def __init__(self, insize, size, has_bias=True, fun=activation.linear):
        """  Constructor for `FeedForward` layer

        Args:
            insize (int):  Size (number of neurons) of layer input.
            size (int):  Size(number of neurons) of layer output.
            has_bias (bool, optional): Whether layer has bias.  If `False`, bias
                is initialised to zero and not trained.
            fun (<function>, optional):  Activation function to apply to output
                of layer.

        """
        super().__init__()
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, size, bias=has_bias)
        self.activation = fun
        self.reset_parameters()

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Returns:
            None:  Parameters for weight and bias of `linear` altered in place.
        """
        winit = orthonormal_matrix(self.size, self.insize)
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        return self.activation(self.linear(x))

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "feed-forward"),
                           ('activation', get_function_name(self.activation)),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        res['params'] = OrderedDict([('W', self.linear.weight)] +
                                    [('b', self.linear.bias)] if self.has_bias else [])
        return res


class Softmax(nn.Module):
    """  (Log) Softmax layer with initial linear transform

         tmp = exp( inmat W + b )
         out = log( row_normalise( tmp ) )

    Attributes:
        insize (int):  Size (number of neurons) expected in layer input.
        size (int):  Size (number of neurons) of layer output.
        has_bias (bool, optional): Whether layer has bias.  If `False`, bias
            is initialised to zero and not trained.
        linear (:nn:`Module`):  Pytorch module implementing linear
            output linear transform.
        activation (:nn:`Module`): Pytorch module implementing log-softmax over
            the final (feature) dimension of input.
    """

    def __init__(self, insize, size, has_bias=True):
        """  Constructor for `Softmax` layer

        Args:
            insize (int):  Size (number of neurons) of layer input.
            size (int):  Size (number of neurons) of layer output.
            has_bias (bool, optional): Whether layer has bias.  If `False`, bias
                is initialised to zero and not trained.
        """
        super().__init__()
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, size, bias=has_bias)
        self.activation = nn.LogSoftmax(2)
        self.reset_parameters()

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Returns:
            None:  Parameters for weight and bias of `linear` altered in place.
        """
        winit = orthonormal_matrix(self.size, self.insize)
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        return self.activation(self.linear(x))

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "softmax"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        res['params'] = OrderedDict([('W', self.linear.weight)] +
                                    [('b', self.linear.bias)] if self.has_bias else [])
        return res


class CudnnGru(nn.Module):
    """   Gated Recurrent Unit compatable with CUDNN

    GRU with "linear_before_reset" formulation used in CUDNN

    Attributes:
        cudnn_gru (:nn:`Module`): Pytorch GRU module
        insize (int):  Size (number of neurons) expected in layer input.
        size (int):  Size (number of neurons) of layer output.
        has_bias (bool, optional): Whether layer has bias.  If `False`, bias
            is initialised to zero and not trained.
    """

    def __init__(self, insize, size, bias=True):
        """  Constructor for `CudnnGru` layer

        Args:
            insize (int):  Size (number of neurons) of layer input.
            size (int):  Size (number of neurons) of layer output.
            has_bias (bool, optional): Whether layer has bias.  If `False`, bias
                is initialised to zero and not trained.
        """
        super().__init__()
        self.cudnn_gru = nn.GRU(insize, size, bias=bias)
        self.insize = insize
        self.size = size
        self.has_bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Returns:
            None:  Parameters for weight and bias of `cudnn_gru` altered
                in-place.
        """
        for name, param in self.named_parameters():
            shape = list(param.shape)
            if 'weight_hh' in name:
                winit = orthonormal_matrix(*shape)
                init_(param, winit)
            elif 'weight_ih' in name:
                winit = orthonormal_matrix(*shape)
                init_(param, winit)
            else:
                init_(param, truncated_normal(shape, sd=0.5))

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        y, hy = self.cudnn_gru.forward(x)
        return y

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "CudnnGru"),
                           ('activation', "tanh"),
                           ('gate', "sigmoid"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias),
                           ('state0', False)])
        iW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_ih_l0)
        sW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_hh_l0)
        ib = _cudnn_to_guppy_gru(self.cudnn_gru.bias_ih_l0)
        sb = _cudnn_to_guppy_gru(self.cudnn_gru.bias_hh_l0)
        res['params'] = OrderedDict([('iW', _reshape(iW, (3, self.size, self.insize))),
                                     ('sW', _reshape(
                                         sW, (3, self.size, self.size))),
                                     ('ib', _reshape(ib, (3, self.size))),
                                     ('sb', _reshape(sb, (3, self.size)))])
        return res


class Lstm(nn.Module):
    """  LSTM layer wrapper around the cudnn LSTM kernel

    See http://colah.github.io/posts/2015-08-Understanding-LSTMs/ for a good
    introduction to LSTMs.

    Attributes:
        lstm (:nn:`Module`): Pytorch LSTM module
        insize (int):  Size (number of neurons) expected in layer input.
        size (int):  Size (number of neurons) of layer output.
        has_bias (bool, optional): Whether layer has bias.  If `False`, bias
            is initialised to zero and not trained.
    """

    def __init__(self, insize, size, has_bias=True):
        """  Constructor for `Lstm` layer

        Args:
            insize (int):  Size (number of neurons) of layer input.
            size (int):  Size (number of neurons) of layer output.
            has_bias (bool, optional): Whether layer has bias.  If `False`, bias
                is initialised to zero and not trained.
        """
        super().__init__()
        self.lstm = nn.LSTM(insize, size, bias=has_bias)
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self._disable_state_bias()
        self.reset_parameters()

    def _disable_state_bias(self):
        """  Disable training of redundant bias parameter and initialise to zero

        Returns:
            None:  Redundant parameter "bias_hh" of `lstm` is set to zero and
                training is disable.
        """
        for name, param in self.lstm.named_parameters():
            if 'bias_hh' in name:
                param.requires_grad = False
                param.zero_()

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Note:
            The redundant bias parameter `bias_hh` is not generated by
        `named_parameters` and so is not initialised.

        Returns:
            None:  Parameters for weight and bias of `lstm` altered in-place.
        """
        for name, param in self.named_parameters():
            shape = list(param.shape)
            if 'weight_hh' in name:
                winit = orthonormal_matrix(*shape)
                init_(param, winit)
            elif 'weight_ih' in name:
                winit = orthonormal_matrix(*shape)
                init_(param, winit)
            else:
                # TODO: initialise forget gate bias to positive value
                init_(param, truncated_normal(shape, sd=0.5))

    def named_parameters(self, prefix='', recurse=True):
        """  Iterator over trainable parameters  of `lstm`

        Args:
            prefix (str, optional):  Prefix for parameters to search.
            recurse (bool, optional):  Whether to recursively descend into
                `lstm` module.

        Yields:
            tuple of str and :nn:`Parameter`: name of `lstm` parameter and the
                parameter.
        """
        for name, param in self.lstm.named_parameters(prefix=prefix, recurse=recurse):
            if 'bias_hh' not in name:
                yield name, param

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        y, hy = self.lstm.forward(x)
        return y

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "LSTM"),
                           ('activation', "tanh"),
                           ('gate', "sigmoid"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        res['params'] = OrderedDict([('iW', _reshape(self.lstm.weight_ih_l0, (4, self.size, self.insize))),
                                     ('sW', _reshape(self.lstm.weight_hh_l0,
                                                     (4, self.size, self.size))),
                                     ('b', _reshape(self.lstm.bias_ih_l0, (4, self.size)))])
        return res


class GruMod(nn.Module):
    """  Gated Recurrent Unit compatable with guppy

    This version of the Gru should be compatable with guppy. It differs from the
    CudnnGru in that the CudnnGru has an additional bias parameter.

    Attributes:
        cudnn_gru (:nn:`Module`): Pytorch GRU module
        insize (int):  Size (number of neurons) expected in layer input.
        size (int):  Size (number of neurons) of layer output.
        has_bias (bool, optional): Whether layer has bias.  If `False`, bias
            is initialised to zero and not trained.
    """

    def __init__(self, insize, size, has_bias=True):
        """  Constructor for `GruMod` layer

        Args:
            insize (int):  Size (number of neurons) of layer input.
            size (int):  Size (number of neurons) of layer output.
            has_bias (bool, optional): Whether layer has bias.  If `False`, bias
                is initialised to zero and not trained.
        """
        super().__init__()
        self.cudnn_gru = nn.GRU(insize, size, bias=has_bias)
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self._disable_state_bias()
        self.reset_parameters()

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Note:
            The redundant bias parameter `bias_hh` is not generated by
        `named_parameters` and so is not initialised.

        Returns:
            None:  Parameters for weight and bias of `cudnn_gru` altered
                in-place.
        """
        for name, param in self.named_parameters():
            shape = list(param.shape)
            if 'weight_hh' in name:
                winit = orthonormal_matrix(*shape)
                init_(param, winit)
            elif 'weight_ih' in name:
                winit = orthonormal_matrix(*shape)
                init_(param, winit)
            else:
                init_(param, truncated_normal(shape, sd=0.5))

    def _disable_state_bias(self):
        """  Disable training of redundant bias parameter and initialise to zero

        Returns:
            None:  Redundant parameter "bias_hh" of `cudnn_gru` is set to zero and
                training is disable.
        """
        for name, param in self.cudnn_gru.named_parameters():
            if 'bias_hh' in name:
                param.requires_grad = False
                param.zero_()

    def named_parameters(self, prefix='', recurse=True):
        """  Iterator over trainable parameters  of `lstm`

        Args:
            prefix (str, optional):  Prefix for parameters to search.
            recurse (bool, optional):  Whether to recursively descend into
                `cudnn_gru` module.

        Yields:
            tuple of str and :nn:`Parameter`: name of `lstm` parameter and the
                parameter.
        """
        prefix = prefix + ('.' if prefix else '')
        for name, param in self.cudnn_gru.named_parameters(recurse=recurse):
            if not 'bias_hh' in name:
                yield prefix + name, param

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        y, hy = self.cudnn_gru.forward(x)
        return y

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "GruMod"),
                           ('activation', "tanh"),
                           ('gate', "sigmoid"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        iW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_ih_l0)
        sW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_hh_l0)
        b = _cudnn_to_guppy_gru(self.cudnn_gru.bias_ih_l0)
        res['params'] = OrderedDict([('iW', _reshape(iW, (3, self.size, self.insize))),
                                     ('sW', _reshape(
                                         sW, (3, self.size, self.size))),
                                     ('b', _reshape(b, (3, self.size)))])
        return res


def _cudnn_to_guppy_gru(p):
    """  Reorder GRU params from CUDNN to Guppy ordering

    The GRU implemented in CUDNN and that implement in Guppy have the gating
    matrix stored in a different order.

    Args:
        p (:torch:`Tensor`): GRU weights in CUDNN order.

    Returns:
        :torch:`Tensor`: GRU weight in Guppy order.
    """
    x, y, z = torch.chunk(p, 3)
    return torch.cat([y, x, z], 0)


class Convolution(nn.Module):
    """1D convolution over the first dimension

    Takes input of shape [time, batch, features] and produces output of shape
    [ceil((time + padding) / stride), batch, features]

    Attributes:
        has_bias (bool): Whether layer has bias.  If `False`, bias is
            initialised to zero and not trained.
        insize (int):  number of features in expected input tensor
        size (int): number of feature in output tensor
        winlen (int): size of window over input
        stride (int): step size between successive windows.
        padding (tuple of int and int): padding applied to start and
            end of input.
        pad (:nn:`Module`):  Pytorch module applying `padding` to time
            dimension of input Tensor.
        conv (:nn:`Module`):  Pytorch module applying 1D convolution to input.
        activation (<function>): function applying activation elementwise to
            result of convolution.
    """

    def __init__(self, insize, size, winlen, stride=1, pad=None,
                 fun=activation.tanh, has_bias=True):
        """  Constructor for `Convolution` layer

        Args:
            insize (int):  number of features in expected input tensor
            size (int): number of feature in output tensor
            winlen (int): size of window over input
            stride (int, optional): step size between successive windows.
                Default stride 1 does not down-sample output.
            pad (tuple of int and int,optional): padding applied to start and
                end of input, or None in which case the padding is set to ensure
                that output length does not depend on `winlen` -- padding is
                (winlen // 2, (winlen - 1) // 2).
            fun (<function>): function applying activation elementwise to
                result of convolution.
            has_bias (bool, optional): Whether layer has bias.  If `False`, bias
                is initialised to zero and not trained.
        """
        super().__init__()
        self.has_bias = has_bias
        self.insize = insize
        self.size = size
        self.stride = stride
        self.winlen = winlen
        if pad is None:
            pad = (winlen // 2, (winlen - 1) // 2)
        self.padding = pad
        self.pad = nn.ConstantPad1d(pad, 0)
        self.conv = nn.Conv1d(kernel_size=winlen, in_channels=insize,
                              out_channels=size, stride=stride, bias=has_bias)
        self.activation = fun
        self.reset_parameters()

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Returns:
            None:  Parameters for weight and bias of `conv` altered in-place.
        """
        winit = orthonormal_matrix(self.conv.weight.shape[0],
                                   np.prod(self.conv.weight.shape[1:]))
        init_(self.conv.weight, winit.reshape(self.conv.weight.shape))
        if self.has_bias:
            binit = truncated_normal(list(self.conv.bias.shape), sd=0.5)
            init_(self.conv.bias, binit)

    def forward(self, x):
        """  Forward method for layer

        Note:
            Tensor is stored in TBF order, but `conv` requires BFT order, so
        the input tensor is wrapped between two permutations.

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        x = x.permute(1, 2, 0)
        out = self.activation(self.conv(self.pad(x)))
        return out.permute(2, 0, 1)

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([("type", "convolution"),
                           ("insize", self.insize),
                           ("size", self.size),
                           ("bias", self.has_bias),
                           ("winlen", self.conv.kernel_size[0]),
                           ("stride", self.conv.stride[0]),
                           ("padding", self.padding),
                           ("activation", get_function_name(self.activation))])
        res['params'] = OrderedDict([("W", self.conv.weight)] +
                                    [("b", self.conv.bias)] if self.has_bias else [])
        return res


class Parallel(nn.Module):
    """  Apply several layers to same input and concatenate results

    Example:
        Simple bidirectional GRU

        >>> Parallel( GRU(size, size), Reverse( GRU(size, size) )

    Attributes:
        sublayers (list of :nn:`Module`): Layers to apply to input tensor.
    """

    def __init__(self, layers):
        """  Constructor for `Parallel` layer

        Args:
            layers (list of :nn:`Module`): Taiyaki layers to apply
        """
        super().__init__()
        self.sublayers = nn.ModuleList(layers)

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        ys = [layer(x) for layer in self.sublayers]
        return torch.cat(ys, 2)

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', "parallel"),
                            ('sublayers', [layer.json()
                                           for layer in self.sublayers])])


class Product(nn.Module):
    """  Element-wise product of list of input layers

    Example:
        Simple gated feed-forward layer

        >>> Product([FeedForward(insize, size, fun=sigmoid),
                     FeedForward(insize, size, fun=linear)])

    Attributes:
        sublayers (list of :nn:`Module`): Layers to apply to input tensor.
    """

    def __init__(self, layers):
        """  Constructor for `Product` layer

        Args:
            layers (list of :nn:`Module`): Taiyaki layers to apply
        """
        super().__init__()
        self.sublayers = nn.ModuleList(layers)

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        ys = self.sublayers[0](x)
        for layer in self.sublayers[1:]:
            ys *= layer(x)
        return ys

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', "Product"),
                            ('sublayers', [layer.json()
                                           for layer in self.sublayers])])


class Serial(nn.Module):
    """  Apply several layers one after the other in Serial fashion


    Attributes:
        sublayers (list of :nn:`Module`): Layers to apply to input tensor.
    """

    def __init__(self, layers):
        """  Constructor for `Serial` layer

        Args:
            layers (list of :nn:`Module`): Taiyaki layers to apply
        """
        super().__init__()
        self.sublayers = nn.ModuleList(layers)

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        for layer in self.sublayers:
            x = layer(x)
        return x

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([
            ('type', "serial"),
            ('sublayers', [layer.json() for layer in self.sublayers])])


class SoftChoice(nn.Module):
    """  Weighted selection from a list of layers

    Apply several layers to input and combine in a weighted fashion.

    Attributes:
        alpha (:nn:`Parameter`): Vector valued weights for layer selection.
        sublayers (list of :nn:`Module`): Layers to choose from.
    """

    def __init__(self, layers):
        """  Constructor for `Serial` layer

        Args:
            layers (list of :nn:`Module`): Taiyaki layers to choose from.
        """
        super().__init__()
        self.sublayers = nn.ModuleList(layers)
        self.alpha = Parameter(torch.zeros(len(layers)))

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        ps = torch.nn.Softmax(0)(self.alpha)
        ys = [p * layer(x) for p, layer in zip(ps, self.sublayers)]
        return torch.stack(ys).sum(0)

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "softchoice"),
                           ('sublayers', [layer.json() for layer in self.sublayers])])
        res['params'] = OrderedDict([('alpha', self.alpha)])
        return res


def zeros(size):
    """  Create a numpy vector zeros with compatible dtype

    Args:
        size (int): length of vector
    """
    return np.zeros(size, dtype=taiyaki_dtype)


def _reshape(x, shape):
    """  Convert a Pytorch tensor to numpy format and reshape

    Args:
        x (:torch:`Tensor`): Pytorch Tensor to reshape
        shape (tuple of int): New shape
    """
    return x.detach_().numpy().reshape(shape)


class Identity(nn.Module):
    """  Apply activation function elementwise

    Attributes:
        fun (<function>): Function that applies an element-wise activation to
            output tensor.
    """

    def __init__(self, fun=activation.linear):
        """  Constructor for `Identity` layer

        Args:
            fun (<function>, optional): A function that applies an element-wise
                activation to output tensor, default `linear` does no
                transformation.
        """
        super().__init__()
        self.fun = fun

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', 'Identity'),
                            ('activation', get_function_name(self.activation))])

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        return self.fun(x)


class Studentise(nn.Module):
    """  Normalize all features in batch

        Studentize features by subtracting mean (over time) and dividing by
    their standard deviation.  A small `espilon` constant is added to the
    standard deviation in order to prevent divide by zero.

    Attributes:
        epsilon (float): Small value to stabilise calculation.
    """

    def __init__(self, epsilon=1e-4):
        """  Constructor for `Studentise` layer

        Args:
            epsilon (float): Small value to stabilise calculation.
        """
        super().__init__()
        self.epsilon = epsilon

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return {'type': "studentise"}

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        features = x.shape[-1]
        m = x.view(-1, features).mean(0)
        v = x.view(-1, features).var(0, unbiased=False)
        return (x - m) / torch.sqrt(v + self.epsilon)


class DeltaSample(nn.Module):
    """  Returns difference between neighbouring features

    Note:
        Right-hand side is padded with zero to maintain input shape.
    """

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', "DeltaSample")])

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        output = x[1:] - x[:-1]
        padding = torch.zeros_like(x[:1])
        return torch.cat((output, padding), dim=0)


class Window(nn.Module):
    """  Create a sliding window over input

    Attributes:
        w (int): Size of window.
    """

    def __init__(self, w):
        """  Constructor for `Window` layer

        Args:
            w (int):  Size of window
        """
        super().__init__()
        assert w > 0, "Window size must be positive"
        assert w % 2 == 1, 'Window size should be odd'
        self.w = w

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "window")])
        res['params'] = OrderedDict([('w', self.w)])
        return res

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        length = x.shape[0]
        pad = self.w // 2
        zeros = x.new_zeros((pad,) + x.shape[1:])
        padded_x = torch.cat([zeros, x, zeros], 0)

        xs = [padded_x[i:length + i] for i in range(self.w)]
        return torch.cat(xs, x.ndimension() - 1)


def birnn(forward, backward):
    """  Creates a bidirectional RNN from two RNNs

    Args:
        forward (:nn:`Module`):  A layer to run forwards
        backward (:nn:`Module`): A layer to run backwards

    Returns:
        :nn:`Module`:  Layer wrapping `forward` and `backward`
    """
    return Parallel([forward, Reverse(backward)])


@torch.jit.script
def logaddexp(x, y):
    """  Element-wise logsumexp

    Performs log( exp(x) + exp(y)) element-wise in a stable manner.

    Args:
        x (:torch:`Tensor`):  Tensor
        y (:torch:`Tensor`):  Tensor, shape must be same as `x`

    Returns:
        :torch:`Tensor`: result of calculation, with same shape as `x`
    """
    return torch.max(x, y) + nn.functional.softplus(-torch.abs(x - y))


@torch.jit.script
def global_norm_flipflop_step(scores_t, fwd_t, nbase):
    """  Single step of flip-flop global normalization

    Args:
        scores_t (:torch:`Tensor`):  Transition scores for time step.
        fwd_t (:torch:`Tensor`): Input state.
        nbase (:torch:`Tensor`): Number of bases

    Returns:
        tuple of :torch:`Tensor` and :torch:`Tensor`: scaling factor for time
            step and new state following time step.
    """
    nbase = int(nbase)
    curr_scores = fwd_t.unsqueeze(1) + scores_t.reshape(
        (-1, nbase + 1, 2 * nbase))
    base1_state = curr_scores[:, :nbase].logsumexp(2)
    base2_state = logaddexp(
        curr_scores[:, nbase, :nbase], curr_scores[:, nbase, nbase:])
    new_state = torch.cat([base1_state, base2_state], dim=1)
    factors = new_state.logsumexp(1, keepdim=True)
    new_state = new_state - factors
    return factors, new_state


def log_partition_flipflop(scores):
    """  Calculate the log of the partition function for flip-flop model

    Args:
        scores (:torch:`Tensor`):  Transition scores for all time steps

    Returns:
        :torch:`Tensor`: log-partition function for each batch.
    """
    T, N, C = scores.shape
    nbase = flipflopfings.nbase_flipflop(C)

    fwd = torch.cat([torch.zeros(N, nbase, device=scores.device,
                                 dtype=scores.dtype),
                     torch.full((N, nbase), -LARGE_LOG_VAL, device=scores.device,
                                dtype=scores.dtype)],
                    1)
    logZ = fwd.logsumexp(1, keepdim=True)
    fwd = fwd - logZ
    nbase = torch.tensor(nbase, device=scores.device, dtype=torch.long)
    for scores_t in scores.unbind(0):
        factors, fwd = global_norm_flipflop_step(scores_t, fwd, nbase)
        logZ = logZ + factors
    return logZ


def global_norm_flipflop(scores):
    """  Globally normalize scores for flip-flop model.

    Args:
        scores (:torch:`Tensor`):  Transition scores for all time steps

    Returns:
        :torch:`Tensor`: scores after global normalization.
    """
    T = scores.shape[0]
    logZ = log_partition_flipflop(scores)
    return scores - logZ / np.float32(T)


class GlobalNormFlipFlop(nn.Module):
    """  Globally normalize scores for flip-flop model.

        Global normalisation is performed after transforming input "x" by
            scale * activation( x W + b)

    Attributes:
        insize (int):  Size (number of features) expected for input tensor
        nbase (int):  Number bases
        size (int):  Size of output tensor
        activation (<function>):  Function that applies elementwise activation
            to output of linear transform (before global normalisation).
        has_bias (bool):  Whether layer has bias.  If `False`, bias is
            initialised to zero and not trained.
        linear (:nn:`Module`):  PyTorch module implementing linear
            transformation.
        scale (float):  Scaling factor to apply.
        _never_use_cupy (bool):  If True, never use accelerated cupy routine
            even if it is available.

    Returns:
        :torch:`Tensor`: scores after global normalization.
    """

    def __init__(self, insize, nbase, has_bias=True, _never_use_cupy=False,
                 fun=activation.tanh, scale=5.0):
        """  Constructor for `GlobalNormFlipFlop` layer

        Args:
            insize (int):  Size (number of features) expected for input tensor
            nbase (int):  Number bases
            has_bias (bool, optional):  Whether layer has bias.  If `False`,
                bias is initialised to zero and not trained.
            _never_use_cupy (bool, optional):  If True, never use accelerated
                cupy routine even if it is available.  Default will use cupy if
                available.
            fun (<function>, optional):  Function that applies elementwise
                activation to output of linear transform (before global
                normalisation).  Default tanh.
            scale (float, optional):  Scaling factor to apply, default 5.0

        """
        super().__init__()
        self.insize = insize
        self.nbase = nbase
        self.size = flipflopfings.nstate_flipflop(nbase)
        self.activation = fun
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, self.size, bias=has_bias)
        self.reset_parameters()
        self.scale = scale
        self._never_use_cupy = _never_use_cupy

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([
            ('type', 'GlobalNormTwoState'),
            ('size', self.size),
            ('insize', self.insize),
            ('bias', self.has_bias),
            ('scale', self.scale),
            ("activation", get_function_name(self.activation))])
        res['params'] = OrderedDict(
            [('W', self.linear.weight)] +
            [('b', self.linear.bias)] if self.has_bias else [])

        return res

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Returns:
            None:  Parameters for weight and bias of `linear` altered in-place.
        """
        winit = orthonormal_matrix(*list(self.linear.weight.shape))
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def _use_cupy(self, x):
        """  Determine whether cupy should be used

        Note:
            getattr used instead of simple look-up for backwards compatibility

        Args:
            x (:torch:`Tensor`): Sample input.

        Returns:
            bool: True if cupy available and `_never_use_cupy` is False.  False
                otherwise.
        """
        # getattr in stead of simple look-up for backwards compatibility
        if self._never_use_cupy:
            return False

        if not x.is_cuda:
            return False

        try:
            from .cupy_extensions import flipflop
            return True
        except ImportError:
            return False

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        y = self.scale * self.activation(self.linear(x))

        if self._use_cupy(x):
            from .cupy_extensions import flipflop
            return flipflop.global_norm(y)
        else:
            return global_norm_flipflop(y)


class GlobalNormFlipFlopCatMod(nn.Module):
    """ Flip-flop layer with additional modified base output stream

    Attributes (can_nmods, output_alphabet and modified_base_long_names) define
    a modified base model and their names and structure is stable.

    The cat_mod model outputs bases in a specific order. This ordering groups
    modified base labels with thier corresponding canonical bases.

    For example alphabet='ACGTZYXW', collapse_alphabet='ACGTCAAT' would produce
    cat_mod ordering of `AYXCZGTW`.

    - `can_mods_offsets` is the offset for each canonical base in the
        cat_mod model output
    - `mod_labels` is the modified base label for each value in alphabet. This
        value is `0` for each canonical base and is incremented by one for each
        modified label conditioned on the canonical base. This is in alphabet
        order and NOT cat_mod order. Using the example alphabet above,
        mod_labels would be `[0, 0, 0, 0, 1, 1, 2, 1]`
    """

    def compute_label_conversions(self):
        """ Compute conversion arrays from input label to canonical base and
        modified training label values

        Returns:
            None: Layer adjusted in-place
        """
        # create table of mod label offsets within each canonical label group
        can_labels, mod_labels = [], []
        can_grouped_mods = dict((can_b, 0) for can_b in self.can_bases)
        for b, can_b in zip(self.alphabet, self.collapse_alphabet):
            can_labels.append(self.can_bases.find(can_b))
            if b in self.can_bases:
                # canonical bases are always 0
                mod_labels.append(0)
            else:
                can_grouped_mods[can_b] += 1
                mod_labels.append(can_grouped_mods[can_b])
        self.can_labels = np.array(can_labels)
        self.mod_labels = np.array(mod_labels)

        return

    def compute_layer_mods_info(self):
        """ Sort alphabet into canonical grouping and rearrange mod_long_names

        Returns:
            None: Layer adjusted in-place
        """
        self.output_alphabet = ''.join(b[1] for b in sorted(zip(
            self.collapse_alphabet, self.alphabet)))
        self.ordered_mod_long_names = (
            None if self.mod_long_names is None else
            [self.mod_name_conv[b] for b in self.alphabet
             if b in self.mod_bases])
        # number of modified bases per canonical base for saving in model file
        # (ordered by self.can_bases)
        self.can_nmods = np.array([
            sum(b == can_b for b in self.collapse_alphabet) - 1
            for can_b in self.can_bases])

        # canonical base offset into flip-flop output layer
        self.can_mods_offsets = np.cumsum(np.concatenate(
            [[0], self.can_nmods + 1])).astype(np.int32)
        # modified base indices from linear layer corresponding to
        # each canonical base
        self.can_indices = []
        curr_n_mods = 0
        for bi_nmods in self.can_nmods:
            # global canonical category is index 0 then mod cat indices
            self.can_indices.append(np.concatenate([
                [0], np.arange(curr_n_mods + 1, curr_n_mods + 1 + bi_nmods)]))
            curr_n_mods += bi_nmods

        return

    def __init__(self, insize, alphabet_info, has_bias=True,
                 _never_use_cupy=False):
        """ Constructor for `GlobalNormFlipFlopCatMod` layer

        Args:
            insize (int): Size (number of features) expected for input tensor
            alphabet_info (:class:`alphabet.AlphabetInfo`): Alphabet for layer,
                containing description of modified bases and their cannonical
                equivalents.
            has_bias (bool, optional): Whether layer has bias. If `False`,
                bias is initialised to zero and not trained.
            _never_use_cupy (bool, optional):  If True, never use accelerated
                cupy routine even if it is available.  Default will use cupy if
                available.
        """
        # IMPORTANT these attributes (can_nmods, output_alphabet and
        # modified_base_long_names) are stable and depended upon by external
        # applications. Their names or data structure should remain stable.
        super().__init__()
        self.insize = insize
        self.has_bias = has_bias
        self._never_use_cupy = _never_use_cupy

        # Extract necessary values from alphabet_info
        self.alphabet = alphabet_info.alphabet
        self.collapse_alphabet = alphabet_info.collapse_alphabet
        self.mod_long_names = alphabet_info.mod_long_names
        self.mod_name_conv = alphabet_info.mod_name_conv
        # can_bases determines the order of the canonical bases in the model
        self.can_bases = alphabet_info.can_bases
        self.mod_bases = alphabet_info.mod_bases
        self.ncan_base = alphabet_info.ncan_base
        self.nmod_base = alphabet_info.nmod_base

        self.compute_label_conversions()
        self.compute_layer_mods_info()

        # standard flip-flop transition states plus (single cat for all bases)
        # canonical and categorical mod states
        self.ntrans_states = 2 * self.ncan_base * (self.ncan_base + 1)
        self.size = self.ntrans_states + 1 + self.nmod_base
        self.lsm = nn.LogSoftmax(2)
        self.linear = nn.Linear(insize, self.size, bias=self.has_bias)
        self.reset_parameters()

        return

    @property
    def nbase(self):
        """ Number of canonical bases

        Returns:
            int: Number of canonical bases
        """
        return self.ncan_base

    def json(self):
        """ Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([
            ('type', 'GlobalNormTwoStateCatMod'),
            ('size', self.size),
            ('insize', self.insize),
            ('bias', self.has_bias),
            ('can_nmods', self.can_nmods),
            ('output_alphabet', self.output_alphabet),
            ('modified_base_long_names', self.ordered_mod_long_names)])
        res['params'] = OrderedDict(
            [('W', self.linear.weight)] +
            [('b', self.linear.bias)] if self.has_bias else [])
        return res

    def reset_parameters(self):
        """ Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Returns:
            None: Parameters for weight and bias of `linear` altered in-place.
        """
        winit = orthonormal_matrix(*list(self.linear.weight.shape))
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def _use_cupy(self, x):
        """ Determine whether cupy should be used

        Note:
            getattr used instead of simple look-up for backwards compatibility

        Args:
            x (:torch:`Tensor`): Sample input.

        Returns:
            bool: True if cupy available and `_never_use_cupy` is False.  False
                otherwise.
        """
        if self._never_use_cupy or not x.is_cuda:
            return False

        if not x.is_cuda:
            return False

        try:
            from .cupy_extensions import flipflop
            return True
        except ImportError:
            return False

    def get_softmax_cat_mods(self, cat_mod_scores):
        """ Get categorical modified base log probabilities from raw neural
        network outputs.

        Note:
            When a base has no associated mods the value is represented
                by a constant Tensor.

        Example:
            Layer that includes mods 5mC, 5hmC and 6mA would take input:
            [Can, A_6mA, C_5mC, C_5hmC]
            and output:
            [A_can, A_6mA, C_can, C_5mC, C_5hmC, G_can, T_can]

        Args:
            cat_mod_scores (:torch:`Tensor`):

        Returns:
            :torch:`Tensor`: Tensor of length ncan_base + nmod
        """
        mod_layers = []
        for lab_indices in self.can_indices:
            mod_layers.append(self.lsm(cat_mod_scores[:, :, lab_indices]))
        return torch.cat(mod_layers, dim=2)

    def forward(self, x):
        """ Forward method for layer

        Args:
            x (:torch:`Tensor`): Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        y = self.linear(x)

        trans_scores = 5.0 * activation.tanh(y[:, :, :self.ntrans_states])
        cat_mod_scores = y[:, :, self.ntrans_states:]
        assert cat_mod_scores.shape[2] == self.nmod_base + 1, (
            'Invalid scores provided to forward:  ' +
            'Expected: {}  got: {}'.format(self.nmod_base + 1,
                                           cat_mod_scores.shape[2]))

        if self._use_cupy(x):
            from .cupy_extensions import flipflop
            norm_trans_scores = flipflop.global_norm(trans_scores)
        else:
            norm_trans_scores = global_norm_flipflop(trans_scores)

        cat_mod_scores = self.get_softmax_cat_mods(cat_mod_scores)
        assert cat_mod_scores.shape[2] == self.nmod_base + self.ncan_base, (
            'Invalid softmax categorical mod scores:  ' +
            'Expected: {}  got: {}'.format(self.nmod_base + self.ncan_base,
                                           cat_mod_scores.shape[2]))

        return torch.cat((norm_trans_scores, cat_mod_scores), dim=2)


def is_cat_mod_model(net):
    """ Is model a categorical modified base model

    Args:
        net (:nn:`Module`):  A Taiyaki network

    Raises:
        AssertionError: Outer later of network is not :class:`Serial`

    Returns:
        bool: True if final layer is categorical mod base, False otherwise
    """
    assert isinstance(net, Serial)
    return isinstance(net.sublayers[-1], GlobalNormFlipFlopCatMod)


class TimeLinear(nn.Module):
    """  Basic feedforward layer applied over time dimension

         out = fun( inMat W + b )

    Attributes:
        insize (int):  Size (number of neurons) expected in layer input.
        size (int):  Size (number of neurons) of layer output.
        has_bias (bool): Whether layer has bias.  If `False`, bias is
            initialised to zero and not trained.
        linear (:nn:`Module`):  Pytorch module implementing linear
            transform.
        activation (<function>, optional):  Activation function to apply to
            output linear transform.
    """

    def __init__(self, insize, size, has_bias=True, fun=activation.linear):
        """  Constructor for `TimeLinear` layer

        Args:
            insize (int):  Size (number of neurons) of layer input.
            size (int):  Size(number of neurons) of layer output.
            has_bias (bool, optional): Whether layer has bias.  If `False`, bias
                is initialised to zero and not trained.
            fun (<function>, optional):  Activation function to apply to output
                of layer.
        """
        super().__init__()
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, size, bias=has_bias)
        self.activation = fun
        self.reset_parameters()

    def reset_parameters(self):
        """  Initialise parameters for layer

        Performs orthogonal initialisation for matrix parameters and truncated
        normal ('Xavier') initialisation for vector parameters.

        Returns:
            None:  Parameters for weight and bias of `linear` altered in-place.
        """
        winit = orthonormal_matrix(self.size, self.insize)
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        xp = x.permute(1, 2, 0)
        y = self.activation(self.linear(xp))
        return y.permute(2, 0, 1)

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        res = OrderedDict([('type', "TimeLinear"),
                           ('activation', get_function_name(self.activation)),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        res['params'] = OrderedDict([('W', self.linear.weight)] +
                                    [('b', self.linear.bias)] if self.has_bias else [])
        return res


class UpSample(nn.Module):
    """  Upsample time by reshaping on time--feature axis

        Upsample time by folding around features.  Reshapes a tensor from
    (nt, nb, nf) to (nt * nfold, nb, nf / nfold)

    Attributes:
        nfold (int): Factor to upsample by.
    """

    def __init__(self, nfold):
        """  Constructor for `UpSample` layer

        Args:
            nfold (int):  Factor to up-sample by.
        """
        super().__init__()
        self.nfold = nfold

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', "UpSample"),
                            ('nfold', self.nfold)])

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        nt, nb, nf = x.shape
        y = x.transpose(1, 0)
        assert nf % self.nfold == 0, "Number of features must be divisible by nfold"
        nf_out = nf // self.nfold
        nt_out = nt * self.nfold
        z = torch.reshape(y, (nb, nt_out, nf_out))
        return z.transpose(1, 0)


class DownSample(nn.Module):
    """  Downsampletime by reshaping on time--feature axis

        Downsample time by concatenating features.  Reshapes a tensor from
    (nt, nb, nf) to (nt / nfold, nb, nf * nfold)

    Attributes:
        nfold (int): Factor to downsample by.
    """

    def __init__(self, nfold):
        """  Constructor for `DownSample` layer

        Args:
            nfold (int):  Factor to down-sample by.
        """
        super().__init__()
        self.nfold = nfold

    def json(self):
        """  Create structured output describing layer for converting to json

        Returns:
            :collections:`OrderedDict`: Structured description of layer.
        """
        return OrderedDict([('type', "DownSample"),
                            ('nfold', self.nfold)])

    def forward(self, x):
        """  Forward method for layer

        Args:
            x (:torch:`Tensor`):  Input to layer

        Returns:
            :torch:`Tensor`: Output of layer
        """
        nt, nb, nf = x.shape
        y = x.transpose(1, 0)
        assert nt % self.nfold == 0, "Number of time points must be divisible by nfold"
        nt_out = nt // self.nfold
        nf_out = nf * self.nfold
        z = torch.reshape(y, (nb, nt_out, nf_out))
        return z.transpose(1, 0)


def DownUpSample(layer, nfold):
    """  Wrap inner layer between down-sampling and up-sampling layers

    For input nt x nb x nf,
        Reshapes to (nt / nfold) x nb x (nf * nfold) == nt2 x nb x nf2
        Runs `layer`  nt2  x nb x nf2 => nt2 x nb x nf3
        Reshapes to nt2 x nb x nf3  => nt x nb x (nf3 / nfold)
    The output size of `layer` should be divisible by nfold

    Args:
        layer (:nn:`Module`):  Layer to wrap
        nfold (int):  N-fold scaling of time

    Returns:
        :nn:`Module`: `taiyaki`:Serial: layer wrapping `layer` between layers
            that down-sample and up-sample by a factor of `nfold`.
    """
    assert layer.size % nfold == 0, "Output of layer not divisible by nfold"
    return Serial([DownSample(nfold), layer, UpSample(nfold)])
