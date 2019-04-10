from taiyaki.activation import tanh
from taiyaki.layers import (
    Convolution, GruMod, Reverse, Serial, GlobalNormFlipFlopCatMod)


def network(insize=1, size=256, winlen=19, stride=2,
            collapse_labels=None, alphabet=None, mod_long_names=None):
    return Serial([
        Convolution(insize, size, winlen, stride=stride, fun=tanh),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GlobalNormFlipFlopCatMod(
            size, collapse_labels, alphabet, mod_long_names),
    ])
