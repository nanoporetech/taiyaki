#!/usr/bin/env python3
import argparse

import torch

from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.cmdargs import FileAbsent, FileExists
from taiyaki import layers


parser = argparse.ArgumentParser(description='Upgrade model file',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

add_common_command_args(parser, ['version'])
parser.add_argument('--output', action=FileAbsent, default=None,
                    help='Name for output upgraded mapped signal file')
parser.add_argument('input', action=FileExists,
                    help='Mapped signal to read from')


def convert_0_to_1(model):
    """ Convert version 0 to version 1
        * Add metadata
        * Convolution model has `has_bias` field
        * GlobalNormFlipFlop has `_never_use_cupy` field
    """
    if hasattr(model, 'metadata'):
        #  Version already at least 1
        return model
    print('Upgrading to version 1')

    #  Add metadata
    model.metadata = {
        'reverse' : False,
        'standardize' : True,
        'version' : 1
    }
    print('Added metadata.  Assumed reads are standardized and not reversed')

    for layer in model.modules():
        #  Walk layers and change
        if isinstance(layer, layers.Convolution):
            print('Checking convolution layer')
            #  Don't assert since field was silently introduced and
            #  some version 0 models have it
            if not hasattr(layer, '_never_use_cupy'):
                layer.has_bias = layer.conv.bias is not None
        if isinstance(layer, layers.GlobalNormFlipFlop):
            print('Checking GlobalNormFlipFlop layer')
            #  Don't assert since field was silently introduced and
            #  some version 0 models have it
            if not hasattr(layer, '_never_use_cupy'):
                layer._never_use_cupy = False


def main():
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input

    print('Loading model from {}'.format(args.input))
    net = torch.load(args.input, map_location=torch.device('cpu'))

    convert_0_to_1(net)

    print('Saving upgraded model to {}'.format(args.output))
    torch.save(net, args.output)



if __name__ == '__main__':
    main()
