import numpy as np

from taiyaki.activation import tanh
from taiyaki.layers import Convolution, GruMod, Reverse, Serial, GlobalNormFlipFlop


def network(insize=1, size=256, winlen=19, stride=2, outsize=40):
    nbase = int(np.sqrt(outsize / 2))

    assert 2 * nbase * (nbase + 1) == outsize,\
        "Invalid size for a flipflop model: nbase = {}, size = {}".format(nbase, outsize)

    return Serial([
        Convolution(insize, size, winlen, stride=stride, fun=tanh),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GlobalNormFlipFlop(size, nbase),
    ])
