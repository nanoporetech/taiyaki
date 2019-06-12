from collections import OrderedDict
import numpy as np

import torch
from torch import nn
from torch.nn import Parameter
from scipy.stats import truncnorm

from taiyaki import activation, flipflopfings
from taiyaki.config import taiyaki_dtype


"""  Convention: inMat row major (C ordering) as (time, batch, state)
"""
_FORGET_BIAS = 2.0


def init_(param, value):
    """Set parameter value (inplace) from tensor, numpy array, list or tuple"""
    value_as_tensor = torch.tensor(value, dtype=param.data.dtype)
    param.data.detach_().set_(value_as_tensor)


def random_orthonormal(n, m=None):
    """  Generate random orthonormal matrix
    :param n: rank of matrix to generate
    :param m: second dimension of matrix, set equal to n if None.

    Distribution may not be uniform over all orthonormal matrices
    (see scipy.stats.ortho_group) but good enough.

    A square matrix is generated if only one parameter is givn, otherwise a
    rectangular matrix is generated.  The number of columns must be greater than
    the number of rows.

    :returns: orthonormal matrix
    """
    m = n if m is None else m
    assert m >= n
    x = np.random.rand(n, m)
    _, _ , Vt = np.linalg.svd(x, full_matrices=False)
    return Vt


def orthonormal_matrix(nrow, ncol):
    """ Generate random orthonormal matrix
    """
    nrep = nrow // ncol
    out = np.zeros((nrow, ncol), dtype='f4')
    for i in range(nrep):
        out[i * ncol : i * ncol + ncol] = random_orthonormal(ncol)
    #  Remaining
    remsize = nrow - nrep * ncol
    if remsize > 0 :
        out[nrep * ncol : , :] = random_orthonormal(remsize, ncol)

    return out


def truncated_normal(size, sd):
    """Truncated normal for Xavier style initiation"""
    res = sd * truncnorm.rvs(-2, 2, size=size)
    return res.astype('f4')


def reverse(x):
    """Reverse input on the first axis"""
    inv_idx = torch.arange(x.size(0) - 1, -1, -1).long()
    if x.is_cuda:
        inv_idx = inv_idx.cuda(x.get_device())
    return x.index_select(0, inv_idx)


class Reverse(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return reverse(self.layer(reverse(x)))

    def json(self, params=False):
        return OrderedDict([('type', "reverse"),
                            ('sublayers', self.layer.json(params))])


class Residual(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return x + self.layer(x)

    def json(self, params=False):
        return OrderedDict([('type', "Residual"),
                            ('sublayers', self.layer.json(params))])


class GatedResidual(nn.Module):
    def __init__(self, layer, gate_init=0.0):
        super().__init__()
        self.layer = layer
        self.alpha = Parameter(torch.tensor([gate_init]))

    def forward(self, x):
        gate = activation.sigmoid(self.alpha)
        y = self.layer(x)
        return gate * x + (1 - gate) * y

    def json(self, params=False):
        res = OrderedDict([('type', "GatedResidual"),
                           ('sublayers', self.layer.json(params))])
        if params:
            res['params'] = OrderedDict([('alpha', float(self.alpha.detach_().numpy()[0]))])
        return res


class FeedForward(nn.Module):
    """  Basic feedforward layer
         out = f( inMat W + b )

    :param insize: Size of input to layer
    :param size: Layer size
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    """
    def __init__(self, insize, size, has_bias=True, fun=activation.linear):
        super().__init__()
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, size, bias=has_bias)
        self.activation = fun
        self.reset_parameters()

    def reset_parameters(self):
        winit = orthonormal_matrix(self.size, self.insize)
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        return self.activation(self.linear(x))

    def json(self, params=False):
        res = OrderedDict([('type', "feed-forward"),
                           ('activation', self.activation.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', self.linear.weight)] +
                                        [('b', self.linear.bias)] if self.has_bias else [])
        return res


class Softmax(nn.Module):
    """  Softmax layer
         tmp = exp( inmat W + b )
         out = log( row_normalise( tmp ) )

    :param insize: Size of input to layer
    :param size: Layer size
    :param has_bias: Whether layer has bias
    """
    def __init__(self, insize, size, has_bias=True):
        super().__init__()
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, size, bias=has_bias)
        self.activation = nn.LogSoftmax(2)
        self.reset_parameters()

    def reset_parameters(self):
        winit = orthonormal_matrix(self.size, self.insize)
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        return self.activation(self.linear(x))

    def json(self, params=False):
        res = OrderedDict([('type', "softmax"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', self.linear.weight)] +
                                        [('b', self.linear.bias)] if self.has_bias else [])
        return res


class CudnnGru(nn.Module):
    """ Gated Recurrent Unit compatable with cudnn

    :param insize: Size of input to layer
    :param size: Layer size
    :param has_bias: Whether layer has bias
    """
    def __init__(self, insize, size, bias=True):
        super().__init__()
        self.cudnn_gru = nn.GRU(insize, size, bias=bias)
        self.insize = insize
        self.size = size
        self.has_bias = bias
        self.reset_parameters()

    def reset_parameters(self):
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
        y, hy = self.cudnn_gru.forward(x)
        return y

    def json(self, params=False):
        res = OrderedDict([('type', "CudnnGru"),
                           ('activation', "tanh"),
                           ('gate', "sigmoid"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias),
                           ('state0', False)])
        if params:
            iW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_ih_l0)
            sW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_hh_l0)
            ib = _cudnn_to_guppy_gru(self.cudnn_gru.bias_ih_l0)
            sb = _cudnn_to_guppy_gru(self.cudnn_gru.bias_hh_l0)
            res['params'] = OrderedDict([('iW', _reshape(iW, (3, self.size, self.insize))),
                                         ('sW', _reshape(sW, (3, self.size, self.size))),
                                         ('ib', _reshape(ib, (3, self.size))),
                                         ('sb', _reshape(sb, (3, self.size)))])
        return res


class Lstm(nn.Module):
    """ LSTM layer wrapper around the cudnn LSTM kernel
    See http://colah.github.io/posts/2015-08-Understanding-LSTMs/ for a good
    introduction to LSTMs.

    Step:
        v = [ input_new, output_old ]
        Pforget = gatefun( v W2 + b2 + state * p1)
        Pupdate = gatefun( v W1 + b1 + state * p0)
        Update  = fun( v W0 + b0 )
        state_new = state_old * Pforget + Update * Pupdate
        Poutput = gatefun( v W3 + b3 + state * p2)
        output_new = fun(state) * Poutput

    :param insize: Size of input to layer
    :param size: Layer size
    :param has_bias: Whether layer has bias
    """
    def __init__(self, insize, size, has_bias=True):
        super().__init__()
        self.lstm = nn.LSTM(insize, size, bias=has_bias)
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self._disable_state_bias()
        self.reset_parameters()

    def _disable_state_bias(self):
        for name, param in self.lstm.named_parameters():
            if 'bias_hh' in name:
                param.requires_grad = False
                param.zero_()

    def reset_parameters(self):
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
        for name, param in self.lstm.named_parameters(prefix=prefix, recurse=recurse):
            if 'bias_hh' not in name:
                yield name, param

    def forward(self, x):
        y, hy = self.lstm.forward(x)
        return y

    def json(self, params=False):
        res = OrderedDict([('type', "LSTM"),
                           ('activation', "tanh"),
                           ('gate', "sigmoid"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('iW', _reshape(self.lstm.weight_ih_l0, (4, self.size, self.insize))),
                                         ('sW', _reshape(self.lstm.weight_hh_l0, (4, self.size, self.size))),
                                         ('b', _reshape(self.lstm.bias_ih_l0, (4, self.size)))])
        return res


class GruMod(nn.Module):
    """ Gated Recurrent Unit compatable with guppy

    This version of the Gru should be compatable with guppy. It differs from the
    CudnnGru in that the CudnnGru has an additional bias parameter.

    :param insize: Size of input to layer
    :param size: Layer size
    :param has_bias: Whether layer has bias
    """
    def __init__(self, insize, size, has_bias=True):
        super().__init__()
        self.cudnn_gru = nn.GRU(insize, size, bias=has_bias)
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self._disable_state_bias()
        self.reset_parameters()

    def reset_parameters(self):
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
        for name, param in self.cudnn_gru.named_parameters():
            if 'bias_hh' in name:
                param.requires_grad = False
                param.zero_()

    def named_parameters(self, prefix='', recurse=True):
        prefix = prefix + ('.' if prefix else '')
        for name, param in self.cudnn_gru.named_parameters(recurse=recurse):
            if not 'bias_hh' in name:
                yield prefix + name, param

    def forward(self, x):
        y, hy = self.cudnn_gru.forward(x)
        return y

    def json(self, params=False):
        res = OrderedDict([('type', "GruMod"),
                           ('activation', "tanh"),
                           ('gate', "sigmoid"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            iW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_ih_l0)
            sW = _cudnn_to_guppy_gru(self.cudnn_gru.weight_hh_l0)
            b = _cudnn_to_guppy_gru(self.cudnn_gru.bias_ih_l0)
            res['params'] = OrderedDict([('iW', _reshape(iW, (3, self.size, self.insize))),
                                         ('sW', _reshape(sW, (3, self.size, self.size))),
                                         ('b', _reshape(b, (3, self.size)))])
        return res


def _cudnn_to_guppy_gru(p):
    """Reorder GRU params from order expected by CUDNN to that required by guppy"""
    x, y, z = torch.chunk(p, 3)
    return torch.cat([y, x, z], 0)


class Convolution(nn.Module):
    """1D convolution over the first dimension

    Takes input of shape [time, batch, features] and produces output of shape
    [ceil((time + padding) / stride), batch, features]

    :param insize: number of features on input
    :param size: number of output features
    :param winlen: size of window over input
    :param stride: step size between successive windows
    :param has_bias: whether layer has bias
    :param fun: the activation function
    :param pad: (int, int) of padding applied to start and end, or None in which
        case the padding used is (winlen // 2, (winlen - 1) // 2) which ensures
        that the output length does not depend on winlen
    """
    def __init__(self, insize, size, winlen, stride=1, pad=None, fun=activation.tanh, has_bias=True):
        super().__init__()
        self.insize = insize
        self.size = size
        self.stride = stride
        self.winlen = winlen
        if pad is None:
            pad = (winlen // 2, (winlen - 1) // 2)
        self.padding = pad
        self.pad = nn.ConstantPad1d(pad, 0)
        self.conv = nn.Conv1d(kernel_size=winlen, in_channels=insize, out_channels=size, stride=stride, bias=has_bias)
        self.activation = fun
        self.reset_parameters()

    def reset_parameters(self):
        winit = orthonormal_matrix(self.conv.weight.shape[0], np.prod(self.conv.weight.shape[1:]))
        init_(self.conv.weight, winit.reshape(self.conv.weight.shape))
        binit = truncated_normal(list(self.conv.bias.shape), sd=0.5)
        init_(self.conv.bias, binit)

    def forward(self, x):
        x = x.permute(1, 2, 0)
        out = self.activation(self.conv(self.pad(x)))
        return out.permute(2, 0, 1)

    def json(self, params=False):
        res = OrderedDict([("type", "convolution"),
                           ("insize", self.insize),
                           ("size", self.size),
                           ("winlen", self.conv.kernel_size[0]),
                           ("stride", self.conv.stride[0]),
                           ("padding", self.padding),
                           ("activation", self.activation.__name__)])
        if params:
            res['params'] = OrderedDict([("W", self.conv.weight),
                                         ("b", self.conv.bias)])
        return res


class Parallel(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.sublayers = nn.ModuleList(layers)

    def forward(self, x):
        ys = [layer(x) for layer in self.sublayers]
        return torch.cat(ys, 2)

    def json(self, params=False):
        return OrderedDict([('type', "parallel"),
                            ('sublayers', [layer.json(params)
                                           for layer in self.sublayers])])


class Product(nn.Module):
    """  Element-wise product of list of input layers

         E.g. Simple gated feed-forward layer
         Product([FeedForward(insize, size, fun=sigmoid), FeedForward(insize, size, fun=linear)])
    """
    def __init__(self, layers):
        super().__init__()
        self.sublayers = nn.ModuleList(layers)

    def forward(self, x):
        ys = self.sublayers[0](x)
        for layer in self.sublayers[1:]:
            ys *= layer(x)
        return ys

    def json(self, params=False):
        return OrderedDict([('type', "Product"),
                            ('sublayers', [layer.json(params)
                                           for layer in self.sublayers])])


class Serial(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.sublayers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.sublayers:
            x = layer(x)
        return x

    def json(self, params=False):
        return OrderedDict([
            ('type', "serial"),
            ('sublayers', [layer.json(params) for layer in self.sublayers])])


class SoftChoice(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.sublayers = nn.ModuleList(layers)
        self.alpha = Parameter(torch.zeros(len(layers)))

    def forward(self, x):
        ps = torch.nn.Softmax(0)(self.alpha)
        ys = [p * layer(x) for p, layer in zip(ps, self.sublayers)]
        return torch.stack(ys).sum(0)

    def json(self, params=False):
        res = OrderedDict([('type', "softchoice"),
                           ('sublayers', [layer.json(params) for layer in self.sublayers])])
        if params:
            res['params'] = OrderedDict([('alpha', self.alpha)])
        return res


def zeros(size):
    return np.zeros(size, dtype=taiyaki_dtype)


def _reshape(x, shape):
    return x.detach_().numpy().reshape(shape)


class Identity(nn.Module):
    """The identity transform"""
    def json(self, params=False):
        return OrderedDict([('type', 'Identity')])

    def forward(self, x):
        return x


class Studentise(nn.Module):
    """ Normal all features in batch

    :param epsilon: Stabilsation layer
    """

    def __init__(self, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon

    def json(self, params=False):
        return {'type' : "studentise"}

    def forward(self, x):
        features = x.shape[-1]
        m = x.view(-1, features).mean(0)
        v = x.view(-1, features).var(0, unbiased=False)
        return (x - m) / torch.sqrt(v + self.epsilon)


class DeltaSample(nn.Module):
    """ Returns difference between neighbouring features

    Right is padded with zero
    """

    def json(self, params=False):
        return OrderedDict([('type', "DeltaSample")])

    def forward(self, x):
        output = x[1:] - x[:-1]
        padding = torch.zeros_like(x[:1])
        return torch.cat((output, padding), dim=0)


class Window(nn.Module):
    """  Create a sliding window over input

    :param w: Size of window
    """

    def __init__(self, w):
        super().__init__()
        assert w > 0, "Window size must be positive"
        assert w % 2 == 1, 'Window size should be odd'
        self.w = w

    def json(self, params=False):
        res = OrderedDict([('type', "window")])
        if params:
            res['params'] = OrderedDict([('w', self.w)])
        return res

    def forward(self, x):
        length = x.shape[0]
        pad = self.w // 2
        zeros = x.new_zeros((pad,) + x.shape[1:])
        padded_x = torch.cat([zeros, x, zeros], 0)

        xs = [padded_x[i:length + i] for i in range(self.w)]
        return torch.cat(xs, x.ndimension() - 1)


def birnn(forward, backward):
    """  Creates a bidirectional RNN from two RNNs

    :param forward: A layer to run forwards
    :param backward: A layer to run backwards
    """
    return Parallel([forward, Reverse(backward)])


class LogAddExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        exp_neg_abs_diff = torch.exp(-torch.abs(x - y))
        z = torch.max(x, y) + torch.log1p(exp_neg_abs_diff)
        ctx.save_for_backward(x, y, z)
        return z

    @staticmethod
    def backward(ctx, outgrad):
        x, y , z = ctx.saved_tensors

        return outgrad * (x - z).exp_(), outgrad * (y - z).exp_()

logaddexp = LogAddExp.apply



def global_norm_flipflop(scores):
    T, N, C = scores.shape
    nbase = flipflopfings.nbase_flipflop(C)

    def step(scores_t, fwd_t):
        curr_scores = fwd_t.unsqueeze(1) + scores_t.reshape(
            (-1, nbase + 1, 2 * nbase))
        base1_state = curr_scores[:, :nbase].logsumexp(2)
        base2_state = logaddexp(
            curr_scores[:, nbase, :nbase], curr_scores[:, nbase, nbase:])
        new_state = torch.cat([base1_state, base2_state], dim=1)
        factors = new_state.logsumexp(1, keepdim=True)
        new_state = new_state - factors
        return factors, new_state

    fwd = scores.new_zeros((N, 2 * nbase))
    logZ = fwd.logsumexp(1, keepdim=True)
    fwd = fwd - logZ
    for scores_t in scores:
        factors, fwd = step(scores_t, fwd)
        logZ = logZ + factors
    return scores - logZ / T


class GlobalNormFlipFlop(nn.Module):
    def __init__(self, insize, nbase, has_bias=True, _never_use_cupy=False):
        super().__init__()
        self.insize = insize
        self.nbase = nbase
        self.size = flipflopfings.nstate_flipflop(nbase)
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, self.size, bias=has_bias)
        self.reset_parameters()
        self._never_use_cupy = _never_use_cupy

    def json(self, params=False):
        res = OrderedDict([
            ('type', 'GlobalNormTwoState'),
            ('size', self.size),
            ('insize', self.insize),
            ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict(
                [('W', self.linear.weight)] +
                [('b', self.linear.bias)] if self.has_bias else [])

        return res

    def reset_parameters(self):
        winit = orthonormal_matrix(*list(self.linear.weight.shape))
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def _use_cupy(self, x):
        # getattr in stead of simple look-up for backwards compatibility
        if getattr(self, '_never_use_cupy', False):
            return False

        if not x.is_cuda:
            return False

        try:
            from .cupy_extensions import flipflop
            return True
        except ImportError:
            return False

    def forward(self, x):
        y = 5.0 * activation.tanh(self.linear(x))

        if self._use_cupy(x):
            from .cupy_extensions import flipflop
            return flipflop.global_norm(y)
        else:
            return global_norm_flipflop(y)


class GlobalNormFlipFlopCatMod(nn.Module):
    """ Flip-flop layer with additional modified base output stream

    :param insize: Size of input to layer (should be 1)
    :param alphabet_info: `taiyaki.alphabet.AlphabetInfo` instance
    :param has_bias: Whether layer has bias
    :param _never_use_cupy: Force use of CPU implementation of flip-flop loss

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
        # sort alphabet into canonical grouping and reagange mod_long_names
        # accordingly
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

    def json(self, params=False):
        res = OrderedDict([
            ('type', 'GlobalNormTwoStateCatMod'),
            ('size', self.size),
            ('insize', self.insize),
            ('bias', self.has_bias),
            ('can_nmods', self.can_nmods),
            ('output_alphabet', self.output_alphabet),
            ('modified_base_long_names', self.ordered_mod_long_names)])
        if params:
            res['params'] = OrderedDict(
                [('W', self.linear.weight)] +
                [('b', self.linear.bias)] if self.has_bias else [])
        return res

    def reset_parameters(self):
        winit = orthonormal_matrix(*list(self.linear.weight.shape))
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def _use_cupy(self, x):
        # getattr in stead of simple look-up for backwards compatibility
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
        """ Get categorical modified base tensors

        Example:
            Layer that includes mods 5mC, 5hmC and 6mA would take input:
            [Can, A_6mA, C_5mC, C_5hmC]
            and output:
            [A_can, A_6mA, C_can, C_5mC, C_5hmC, G_can, T_can]

        Outout length is (4 + nmod)

        When a base has no associated mods the value is represented
        by a constant Tensor
        """
        mod_layers = []
        for lab_indices in self.can_indices:
            mod_layers.append(self.lsm(cat_mod_scores[:,:,lab_indices]))
        return torch.cat(mod_layers, dim=2)

    def forward(self, x):
        y = self.linear(x)

        trans_scores = 5.0 * activation.tanh(y[:,:,:self.ntrans_states])
        cat_mod_scores = y[:,:,self.ntrans_states:]
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


class TimeLinear(nn.Module):
    """  Basic feedforward layer over time dimension
         out = f( inMat W + b )

    :param insize: Size of input to layer
    :param size: Layer size
    :param has_bias: Whether layer has bias
    :param fun: The activation function.
    """
    def __init__(self, insize, size, has_bias=True, fun=activation.linear):
        super().__init__()
        self.insize = insize
        self.size = size
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, size, bias=has_bias)
        self.activation = fun
        self.reset_parameters()

    def reset_parameters(self):
        winit = orthonormal_matrix(self.size, self.insize)
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        xp = x.permute(1, 2, 0)
        y = self.activation(self.linear(xp))
        return y.permute(2, 0, 1)

    def json(self, params=False):
        res = OrderedDict([('type', "TimeLinear"),
                           ('activation', self.activation.__name__),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', self.linear.weight)] +
                                        [('b', self.linear.bias)] if self.has_bias else [])
        return res


def rle_log_partition(weights):
    ntime, nbatch, nstate = weights.shape
    nbase = nstate // 2
    hole = 1 - torch.eye(nbase, device=weights.device, dtype=weights.dtype)

    def step(w, ft):
        move = w[:, :4]
        stay = w[:, 4:]
        state_sum = torch.logsumexp(ft.unsqueeze(1) * hole, 2)
        stay_score = ft + stay
        move_score = state_sum + move
        return logaddexp(stay_score, move_score)

    fwd = weights.new_zeros((nbatch, nbase))
    for w_t in weights:
        fwd = step(w_t, fwd)
    return fwd.logsumexp(1, keepdim=True)


class GlobalNormRLE(nn.Module):
    def __init__(self, insize, nbase, has_bias=True):
        super().__init__()
        self.insize = insize
        self.nbase = nbase
        self.size = 4 * nbase
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, self.size, bias=has_bias)
        self.reset_parameters()

    def reset_parameters(self):
        winit = orthonormal_matrix(self.size, self.insize)
        init_(self.linear.weight, winit)
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        y = self.linear(x)
        ntime , _ , _ = y.shape
        shape = 1.0 + activation.softplus(y[... , :self.nbase])
        scale = 0.1 + activation.softplus(y[... , self.nbase : 2 * self.nbase])
        trans = 5.0 * activation.tanh(y[... , 2 * self.nbase:])
        logZ = rle_log_partition(trans).unsqueeze(0) / ntime
        return torch.cat([shape, scale, trans - logZ], dim=2)

    def json(self, params=False):
        res = OrderedDict([('type', "GlobalNormRLE"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', self.linear.weight)] +
                                        [('b', self.linear.bias)] if self.has_bias else [])
        return res


def crfrle_log_partition(weights, nbase):
    ntime, nbatch, nstate = weights.shape
    assert 2 * nbase * nbase == nstate, "Number of states inconsistent with number of states"
    hole = torch.eye(nbase, device=weights.device, dtype=weights.dtype)
    holes = -1e30 * torch.cat([hole, hole], dim=1)
    diag = -1e30 * (1 - torch.cat([hole, hole], dim=1))

    def step(w, ft):
        w_mat = w.reshape((-1, nbase, 2 * nbase))
        ft_big = ft.unsqueeze(1)
        move_score = torch.logsumexp(ft_big + w_mat + holes, 2)
        stay_score = torch.logsumexp(ft_big + w_mat + diag, 2)
        return torch.cat([move_score, stay_score], 1)

    fwd = weights.new_zeros((nbatch, 2 * nbase))
    for w_t in weights:
        fwd = step(w_t, fwd)
    return fwd.logsumexp(1, keepdim=True)


class GlobalNormCRFRLE(nn.Module):
    def __init__(self, insize, nbase, has_bias=True):
        super().__init__()
        self.insize = insize
        self.nbase = nbase
        self.size = nbase * (2 + 2 * nbase)
        self.has_bias = has_bias
        self.linear = nn.Linear(insize, self.size, bias=has_bias)
        self.reset_parameters()

    def reset_parameters(self):
        winit = truncated_normal(list(self.linear.weight.shape), sd=0.5)
        init_(self.linear.weight, winit / np.sqrt(self.insize + self.size))
        if self.has_bias:
            binit = truncated_normal(list(self.linear.bias.shape), sd=0.5)
            init_(self.linear.bias, binit)

    def forward(self, x):
        y = self.linear(x)
        ntime , _ , _ = y.shape
        shape = 1.0 + activation.softplus(y[... , :self.nbase])
        scale = 0.1 + activation.softplus(y[... , self.nbase : 2 * self.nbase])
        trans = 5.0 * activation.tanh(y[... , 2 * self.nbase:])
        logZ = crfrle_log_partition(trans, self.nbase).unsqueeze(0) / ntime
        return torch.cat([shape, scale, trans - logZ], dim=2)

    def json(self, params=False):
        res = OrderedDict([('type', "GlobalNormCRFRLE"),
                           ('size', self.size),
                           ('insize', self.insize),
                           ('bias', self.has_bias)])
        if params:
            res['params'] = OrderedDict([('W', self.linear.weight)] + [('b', self.linear.bias)] if self.has_bias else [])
        return res
