#!/usr/bin/env python3
import argparse

import torch

from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.cmdargs import FileAbsent, FileExists
from taiyaki import activation, layers


parser = argparse.ArgumentParser(description='Upgrade model file',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_common_command_args(parser, ['version'])
parser.add_argument('--output', action=FileAbsent, default=None,
                    help='Name for output upgraded mapped signal file')
parser.add_argument('input', action=FileExists,
                    help='Mapped signal to read from')


def convert_0_to_1(model):
    """ Converts model from version 0 to version 1

    - Adds model metadata (with default values)
    - Adds `has_bias` field to Convolution layers
    - Adds `_never_use_cupy` field to GlobalNormFlipFlop layers
    """
    if hasattr(model, 'metadata'):
        #  Version already at least 1
        return False
    print('Upgrading to version 1')

    #  Add metadata
    model.metadata = {
        'reverse': False,
        'standardize': True,
        'version': 1
    }
    print('Added metadata.  Assumed reads are standardized and not reversed')

    for layer in model.modules():
        #  Walk layers and change
        if isinstance(layer, layers.Convolution):
            print('Checking convolution layer')
            #  Don't assert since field was silently introduced and
            #  some version 0 models have it
            if not hasattr(layer, 'has_bias'):
                layer.has_bias = layer.conv.bias is not None
        if isinstance(layer, layers.GlobalNormFlipFlop):
            print('Checking GlobalNormFlipFlop layer')
            #  Don't assert since field was silently introduced and
            #  some version 0 models have it
            if not hasattr(layer, '_never_use_cupy'):
                layer._never_use_cupy = False
    return True


def convert_1_to_2(model):
    """ Converts model from version 1 to version 2

    - Adds `activation` and `scale` fields to GlobalNormFlipFlop layers
    """
    if model.metadata['version'] >= 2:
        return False
    print('Upgrading to version 2')
    model.metadata['version'] = 2

    for layer in model.modules():
        #  Walk layers and change
        if isinstance(layer, layers.GlobalNormFlipFlop):
            print('Adding activation (tanh) and scale (5.0) to GlobalNormFlipFlop')
            assert not hasattr(layer, 'activation'), 'Inconsistent model!'
            layer.activation = activation.tanh
            assert not hasattr(layer, 'scale'), 'Inconsistent model!'
            layer.scale = 5.0
    return True


def convert_2_to_3(model):
    """ Converts model from version 2 to version 3

    - Add `_flat_weights_names` to `torch.nn.GRU/LSTM` for compatibility with
      torch >= 1.4
    """
    if model.metadata['version'] >= 3:
        return False
    print('Upgrading to version 3')
    model.metadata['version'] = 3

    for layer in model.modules():
        #  Walk layers and change
        if isinstance(layer, torch.nn.LSTM):
            print('Adding _flat_weights_names to torch.nn.LSTM')
            layer._flat_weights_names = layer._all_weights[0]
        if isinstance(layer, torch.nn.GRU):
            print('Adding _flat_weights_names to torch.nn.GRU')
            layer._flat_weights_names = layer._all_weights[0]
    return True


def main():
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input

    print('Loading model from {}'.format(args.input))
    net = torch.load(args.input, map_location=torch.device('cpu'))

    upgraded = False
    upgraded |= convert_0_to_1(net)
    upgraded |= convert_1_to_2(net)
    upgraded |= convert_2_to_3(net)

    if upgraded:
        print('Saving upgraded model to {}'.format(args.output))
        torch.save(net, args.output)


if __name__ == '__main__':
    main()
