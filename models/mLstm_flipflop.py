from taiyaki.flipflopfings import nbase_flipflop
from taiyaki.activation import swish, tanh
from taiyaki.layers import Convolution, Lstm, Reverse, Serial, GlobalNormFlipFlop

"""   More initial processing, drop stride
"""
def network(insize=1, size=256, winlen=19, stride=5, alphabet_info=None):
    nbase = 4 if alphabet_info is None else alphabet_info.nbase
    winlen2 = 5

    return Serial([
        Convolution(insize, 4, winlen2, stride=1, fun=swish),
        Convolution(4, 16, winlen2, stride=1, fun=swish),
        Convolution(16, size, winlen, stride=stride, fun=swish),
        Reverse(Lstm(size, size)),
        Lstm(size, size),
        Reverse(Lstm(size, size)),
        Lstm(size, size),
        Reverse(Lstm(size, size)),
        GlobalNormFlipFlop(size, nbase),
    ])
