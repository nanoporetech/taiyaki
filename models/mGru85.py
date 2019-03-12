from taiyaki.activation import tanh
from taiyaki.layers import Convolution, GruMod, Reverse, Serial, Softmax


def network(insize=1, size=85, winlen=19, stride=5, outsize=1025):
    return Serial([
        Convolution(insize, size, winlen, stride=stride, fun=tanh),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        GruMod(size, size),
        Reverse(GruMod(size, size)),
        Softmax(size, outsize),
    ])
