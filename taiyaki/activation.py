import torch
#  Some activation functions
#  Many based on M-estimations functions, see
#  http://research.microsoft.com/en-us/um/people/zhang/INRIA/Publis/Tutorial-Estim/node24.html


#  Unbounded
def sqr(x):
    # See https://github.com/pytorch/pytorch/issues/2618
    return torch.pow(x, 2)


def linear(x):
    return x


def relu(x):
    return torch.relu(x)


def relu_smooth(x):
    y = torch.clamp(x, 0.0, 1.0)
    return sqr(y) - 2.0 * y + x + abs(x)


def softplus(x):
    """  Softplus function log(1 + exp(x))

        Calculated in a way stable to large and small values of x.  The version
        of this routine in theano.tensor.nnet clips the range of x, potential
        causing NaN's to occur in the softmax (all inputs clipped to zero).

        x >=0  -->  x + log1p(exp(-x))
        x < 0  -->  log1p(exp(x))

        This is equivalent to relu(x) + log1p(exp(-|x|))
    """
    absx = abs(x)
    softplus_neg = torch.log1p(torch.exp(-absx))
    return relu(x) + softplus_neg


def elu(x, alpha=1.0):
    """  Exponential Linear Unit
         See "Fast and Accuracte Deep Network Learning By Exponential Linear
         Units" Clevert, Unterthiner and Hochreiter.
         https://arxiv.org/pdf/1511.07289.pdf

         :param alpha: Exponential scaling parameter, see paper for details.
    """
    return selu(x, alpha, 1.0)


def selu(x, alpha=1.6733, lam=1.0507):
    """  Scaled Exponential Linear Unit
         See "Self-Normalizing Neural Networks" Klambauer, Unterthiner, Mayr
         and Hocreiter.  https://arxiv.org/pdf/1706.02515.pdf

         :param alpha: Exponential scaling parameter, see paper for details.
         :param lam: Scaling parameter, see paper for details.
    """
    return lam * torch.where(x > 0, x, alpha * torch.expm1(x))


def gelu(x):
    """  Gaussian Error Linear Unit

         https://arxiv.org/pdf/1606.08415.pdf

         Using approximation from paper above.

         .. Note::  Why isn't this approximation :math:`x \sigma(1.813799 x)`,
         which would be the result of replacing the Gaussian distribution
         function with a Logistic distribution with unit variance?

    """
    #return 0.5 * (1.0 + torch.tanh(x * 0.7978846 * (1.0 + 0.044715 * x * x)))
    return x * torch.sigmoid(1.702 * x)


def exp(x):
    return torch.exp(x)


def swish(x):
    """ Swish activation

        Swish is self-gated linear activation :math:`x \sigma(x)`
        Original definition has a scaling parameter for the gating value,
        making it a generalisation of the (logistic approximation to) the GELU.
        Evidence presented, e.g. https://arxiv.org/abs/1908.08681 that swish-1
        performs comparable to tuning the parameter.

        https://arxiv.org/abs/1710.05941

    """
    return x * torch.sigmoid(x)


#  Bounded and monotonic


def tanh(x):
    return torch.tanh(x)


def sigmoid(x):
    return torch.sigmoid(x)


def erf(x):
    return torch.erf(x)


def L1mL2(x):
    return x / torch.sqrt(1.0 + 0.5 * x * x)


def fair(x):
    return x / (1.0 + abs(x) / 1.3998)


def retu(x):
    """ Rectifying activation followed by Tanh

    Inspired by more biological neural activation, see figure 1
    http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf
    """
    return tanh(relu(x))


def tanh_pm(x):
    """  Poor man's tanh
    Linear approximation by tangent at x=0.  Clip into valid range.
    """
    return torch.clamp(x, -1.0, 1.0)


def sigmoid_pm(x):
    """ Poor man's sigmoid
    Linear approximation by tangent at x=0.  Clip into valid range.
    """
    return torch.clamp(0.5 + 0.25 * x, 0.0, 1.0)


def bounded_linear(x):
    """ Linear activation clipped into -1, 1
    """
    return torch.clamp(x, -1.0, 1.0)


#  Bounded and redescending
def sin(x):
    return torch.sin(x)


def cauchy(x):
    return x / (1.0 + sqr(x / 2.3849))


def geman_mcclure(x):
    return x / sqr(1.0 + sqr(x))


def welsh(x):
    return x * exp(-sqr(x / 2.9846))
