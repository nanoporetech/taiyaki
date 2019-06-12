from taiyaki.activation import tanh
from taiyaki.layers import Convolution, GlobalNormCRFRLE, GruMod, Reverse, Serial


def network(insize=1, size=256, winlen=19, stride=2, outsize=8):
    nbase = 4
    return Serial([
        Convolution(insize, size, winlen, stride=stride, fun=tanh),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GlobalNormCRFRLE(size, nbase),
    ])
