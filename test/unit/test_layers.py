import abc
import pickle
import json
import numpy as np
import tempfile
import torch
import unittest

from taiyaki import activation
from taiyaki.config import taiyaki_dtype, torch_dtype, numpy_dtype
from taiyaki.json import JsonEncoder
import taiyaki.layers as nn


def rvs(dim):
    '''
    Draw random samples from SO(N)

    Taken from

    http://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
    '''
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim,))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1,))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = (np.eye(dim - n + 1) - 2. * np.outer(x, x) / (x * x).sum())
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1) ** (1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


class ANNTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        np.random.seed(0xdeadbeef)
        self._NSTEP = 25
        self._NFEATURES = 3
        self._SIZE = 64
        self._NBATCH = 2

        self.W = np.random.normal(size=(self._SIZE, self._NFEATURES)).astype(numpy_dtype)
        self.b = np.random.normal(size=self._SIZE).astype(numpy_dtype)
        self.x = np.random.normal(size=(self._NSTEP, self._NBATCH, self._NFEATURES)).astype(numpy_dtype)
        self.res = self.x.dot(self.W.transpose()) + self.b

    def test_000_single_layer_linear(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                                 fun=activation.linear)
        nn.init_(network.linear.weight, self.W)
        nn.init_(network.linear.bias, self.b)
        with torch.no_grad():
            y = network(torch.tensor(self.x)).numpy()
        np.testing.assert_almost_equal(y, self.res, decimal=5)

    def test_001_single_layer_tanh(self):
        network = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                                 fun=activation.tanh)
        nn.init_(network.linear.weight, self.W)
        nn.init_(network.linear.bias, self.b)
        with torch.no_grad():
            y = network(torch.tensor(self.x)).numpy()
        np.testing.assert_almost_equal(y, np.tanh(self.res), decimal=5)

    def test_002_parallel_layers(self):
        l1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        nn.init_(l1.linear.weight, self.W)
        nn.init_(l1.linear.bias, self.b)
        l2 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        nn.init_(l2.linear.weight, self.W)
        nn.init_(l2.linear.bias, self.b)
        network = nn.Parallel([l1, l2])

        with torch.no_grad():
            res = network(torch.tensor(self.x)).numpy()
        np.testing.assert_almost_equal(res[:, :, :self._SIZE], res[:, :, self._SIZE:])

    def test_003_simple_serial(self):
        W2 = np.random.normal(size=(self._SIZE, self._SIZE)).astype(taiyaki_dtype)
        res = self.res.dot(W2.transpose())

        l1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True,
                            fun=activation.linear)
        nn.init_(l1.linear.weight, self.W)
        nn.init_(l1.linear.bias, self.b)
        l2 = nn.FeedForward(self._SIZE, self._SIZE, fun=activation.linear, has_bias=False)
        nn.init_(l2.linear.weight, W2)
        network = nn.Serial([l1, l2])

        with torch.no_grad():
            y = network(torch.tensor(self.x)).numpy()
        np.testing.assert_almost_equal(y, res, decimal=4)

    def test_004_reverse(self):
        network1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        nn.init_(network1.linear.weight, self.W)
        nn.init_(network1.linear.bias, self.b)
        network2 = nn.Reverse(network1)
        with torch.no_grad():
            res1 = network1(torch.tensor(self.x)).numpy()
            res2 = network2(torch.tensor(self.x)).numpy()

        np.testing.assert_almost_equal(res1, res2, decimal=5)

    def test_005_poormans_birnn(self):
        layer1 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        nn.init_(layer1.linear.weight, self.W)
        nn.init_(layer1.linear.bias, self.b)
        layer2 = nn.FeedForward(self._NFEATURES, self._SIZE, has_bias=True)
        nn.init_(layer2.linear.weight, self.W)
        nn.init_(layer2.linear.bias, self.b)
        network = nn.birnn(layer1, layer2)

        with torch.no_grad():
            res = network(torch.tensor(self.x)).numpy()
        np.testing.assert_almost_equal(res[:, :, :self._SIZE], res[:, :, self._SIZE:], decimal=5)

    def test_006_softmax(self):
        network = nn.Softmax(self._NFEATURES, self._SIZE, has_bias=True)

        with torch.no_grad():
            res = network(torch.tensor(self.x)).numpy()
            res_sum = np.exp(res).sum(axis=2)
        self.assertTrue(np.allclose(res_sum, 1.0))

    def test_016_window(self):
        _WINLEN = 3
        network = nn.Window(_WINLEN)
        with torch.no_grad():
            res = network(torch.tensor(self.x)).numpy()
        #  Window is now 'SAME' not 'VALID'. Trim
        wh = _WINLEN // 2
        res = res[wh: -wh]
        for j in range(self._NBATCH):
            for i in range(_WINLEN - 1):
                try:
                    np.testing.assert_almost_equal(
                        res[:, j, i * _WINLEN: (i + 1) * _WINLEN], self.x[i: 1 + i - _WINLEN, j])
                except:
                    win_max = np.amax(np.fabs(res[:, :, i * _WINLEN: (i + 1) * _WINLEN] - self.x[i: 1 + i - _WINLEN]))
                    print("Window max: {}".format(win_max))
                    raise
            np.testing.assert_almost_equal(res[:, j, _WINLEN * (_WINLEN - 1):], self.x[_WINLEN - 1:, j])
            #  Test first and last rows explicitly
            np.testing.assert_almost_equal(self.x[:_WINLEN, j].ravel(), res[0, j].transpose().ravel())
            np.testing.assert_almost_equal(self.x[-_WINLEN:, j].ravel(), res[-1, j].transpose().ravel())

    @unittest.skip('Decoding needs fixing')
    def test_017_decode_simple(self):
        _KMERLEN = 3
        network = nn.Decode(_KMERLEN)
        f = network.compile()
        res = f(self.res)

    def test_018_studentise(self):
        network = nn.Studentise()
        with torch.no_grad():
            res = network(torch.tensor(self.x)).numpy()

        np.testing.assert_almost_equal(np.mean(res, axis=(0, 1)), 0.0)
        np.testing.assert_almost_equal(np.std(res, axis=(0, 1)), 1.0, decimal=4)

    def test_019_identity(self):
        network = nn.Identity()
        res = network(torch.tensor(self.res)).numpy()

        np.testing.assert_almost_equal(res, self.res)


class LayerTest(metaclass=abc.ABCMeta):
    """Mixin abstract class for testing basic layer functionality
    Writing a TestCase for a new layer is easy, for example:

    class RecurrentTest(LayerTest, unittest.TestCase):
        # Inputs for testing the Layer.run() method
        _INPUTS = [np.zeros((10, 20, 12)),
                   np.random.uniform(size=(10, 20, 12)),]

        # The setUp method should instantiate the layer
        def setUp(self):
            self.layer = nn.Recurrent(12, 64)
    """

    _INPUTS = None  # List of input matrices for testing the layer's run method

    @abc.abstractmethod
    def setUp(self):
        """Create the layer as self.layer"""
        return

    def test_000_run(self):
        if self._INPUTS is None:
            raise NotImplementedError("Please specify layer inputs for testing, or explicitly skip this test.")
        f = self.layer.train(False)
        outs = [f(torch.tensor(x, dtype=torch_dtype)) for x in self._INPUTS]

    def test_001_train(self):
        if self._INPUTS is None:
            raise NotImplementedError("Please specify layer inputs for testing, or explicitly skip this test.")
        f = self.layer.train(True)
        outs = [f(torch.tensor(x, dtype=torch_dtype)) for x in self._INPUTS]

    def test_002_json_dumps(self):
        js = json.dumps(self.layer.json(), cls=JsonEncoder)
        js2 = json.dumps(self.layer.json(params=True), cls=JsonEncoder)

    def test_003_json_decodes(self):
        props = json.JSONDecoder().decode(json.dumps(self.layer.json(), cls=JsonEncoder))
        props2 = json.JSONDecoder().decode(json.dumps(self.layer.json(params=True), cls=JsonEncoder))


class LstmTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]

    def setUp(self):
        self.layer = nn.Lstm(12, 64)


class GruModTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.zeros((10, 20, 12)),
               np.random.uniform(size=(10, 20, 12)), ]

    def setUp(self):
        self.layer = nn.GruMod(12, 64)


class ConvolutionTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]

    def setUp(self):
        self.layer = nn.Convolution(12, 32, 11, 5, has_bias=True)


class ResidualTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]

    def setUp(self):
        sublayer = nn.FeedForward(12, 12, has_bias=True)
        self.layer = nn.Residual(sublayer)


class DeltaSampleTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]

    def setUp(self):
        self.layer = nn.DeltaSample()


class ProductTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]

    def setUp(self):
        self.linear1 = nn.FeedForward(12, 8)
        self.linear2 = nn.FeedForward(12, 8)
        self.layer = nn.Product([self.linear1, self.linear2])

    def test_Product_gives_same_result_as_product(self):
        for x in self._INPUTS:
            xt = torch.tensor(x, dtype=torch_dtype)
            with torch.no_grad():
                y1 = self.linear1(xt)
                y2 = self.linear2(xt)
                y = self.layer(xt)
                max_error = (y - y1 * y2).abs().max().float()
            self.assertAlmostEqual(max_error, 0.0)


class TimeLinearTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]

    def setUp(self):
        self.layer = nn.TimeLinear(100, 2)


class GlobalNormFlipFlopTest(LayerTest, unittest.TestCase):
    _INPUTS = [np.random.uniform(size=(100, 20, 12))]

    def setUp(self):
        self.layer = nn.GlobalNormFlipFlop(12, 4)

    @unittest.skip("Test requires GPU")
    def test_cupy_and_non_cupy_same(self):
        layer = nn.GlobalNormFlipFlop(12, 4).cuda()

        # Perform calculation using cupy
        x1 = torch.randn((100, 4, 12)).cuda()
        x1.requires_grad = True
        loss1 = layer(x1).sum()
        loss1.backward()

        # Repeat calculation using pure pytorch
        x2 = x1.detach()
        x2.requires_grad = True
        layer._never_use_cupy = True
        loss2 = layer(x2).sum()
        loss2.backward()

        # Results and gradients should match
        self.assertTrue(torch.allclose(loss1, loss2))
        # Higher atol on gradient because the final operation is a softmax, and
        # rtol before softmax = atol after softmax. Therefore I've replaced
        # the atol with the default value for rtol.
        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=1e-05))
