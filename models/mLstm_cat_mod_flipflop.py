from taiyaki.activation import swish
from taiyaki.layers import (
    Convolution, Lstm, Reverse, Serial, GlobalNormFlipFlopCatMod)


def network(insize=1, size=256, winlen=19, stride=5, alphabet_info=None):
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
        GlobalNormFlipFlopCatMod(size, alphabet_info),
    ])
