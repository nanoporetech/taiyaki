from taiyaki.flipflopfings import nbase_flipflop
from taiyaki.activation import tanh
from taiyaki.layers import Convolution, GruMod, Reverse, Serial, GlobalNormFlipFlop


def network(insize=1, size=256, winlen=19, stride=2, outsize=40):
    nbase = nbase_flipflop(outsize)

    return Serial([
        Convolution(insize, size, winlen, stride=stride, fun=tanh),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GlobalNormFlipFlop(size, nbase),
    ])
