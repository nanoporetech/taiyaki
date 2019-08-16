#!/usr/bin/env python3
import argparse
from collections import OrderedDict
import json
import re
import sys

import numpy as np
import torch

from taiyaki import alphabet
from taiyaki.activation import tanh
from taiyaki.cmdargs import FileExists
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.flipflopfings import nbase_flipflop
from taiyaki.layers import (
    Convolution, GruMod, Reverse, Serial, GlobalNormFlipFlop,
    GlobalNormFlipFlopCatMod)


COMPATIBLE_LAYERS = set((
    'convolution',
    'GruMod',
    'reverse',
    'GlobalNormTwoState',
    'GlobalNormTwoStateCatMod'))


parser = argparse.ArgumentParser(
    description='Convert JSON representation of model to pytorch checkpoint ' +
    'for use within taiyaki/megalodon.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_common_command_args(
    parser, ["output"])

parser.add_argument(
    'json_model', action=FileExists, help='JSON model with params')

def set_params(layer, jsn_params):
    params_od = OrderedDict()
    for layer_name, layer_params in layer.state_dict().items():
        # match layer names (see taiyaki.layer.[layer_type].json functions)
        if re.search('weight_ih', layer_name) and 'iW' in jsn_params:
            # For gru layers convert from guppy format back to pytorch format
            jsn_layer_params = torch.Tensor(np.concatenate([
                jsn_params['iW'][1], jsn_params['iW'][0], jsn_params['iW'][2]]))
        elif re.search('weight_hh', layer_name) and 'sW' in jsn_params:
            # For gru layers convert from guppy format back to pytorch format
            jsn_layer_params = torch.Tensor(np.concatenate([
                jsn_params['sW'][1], jsn_params['sW'][0], jsn_params['sW'][2]]))
        elif re.search('bias_ih', layer_name) and 'b' in jsn_params:
            # For gru layers convert from guppy format back to pytorch format
            jsn_layer_params = torch.Tensor(np.concatenate([
                jsn_params['b'][1], jsn_params['b'][0], jsn_params['b'][2]]))
        elif re.search('bias_hh', layer_name):
            # bias_hh layer not actually used
            jsn_layer_params = torch.zeros_like(layer_params)
        elif re.search('weight', layer_name) and 'W' in jsn_params:
            jsn_layer_params = torch.Tensor(np.array(jsn_params['W']))
        elif re.search('bias', layer_name) and 'b' in jsn_params:
            jsn_layer_params = torch.Tensor(np.array(jsn_params['b']))
        elif layer_name in jsn_params:
            jsn_layer_params = torch.Tensor(np.array(jsn_params[layer_name]))
        else:
            sys.stderr.write((
                'Incompatible layer parameter type ' +
                '({}) encountered.\n').format(layer_name))
            sys.exit(1)
        # TODO could add additional checks for layer size or names, but
        # this covers the applicable layers in the current release.

        params_od[layer_name] = jsn_layer_params

    # set state_dict via OrderedDict of numpy arrays
    layer.load_state_dict(params_od, strict=True)

    return layer

def parse_sublayer(sublayer):
    # TODO apply additional attributes (e.g. has_bias, convolutional padding)
    if sublayer['type'] == 'convolution':
        if sublayer['activation'] != 'tanh':
            sys.stderr.write((
                'Incompatible convolutional layer activation fucntion ' +
                '({}) encountered.\n').format(sublayer['type']))
            sys.exit(1)
        sys.stderr.write((
            'Loading convolutional layer with attributes:\n\tin size: {}\n' +
            '\tout size: {}\n\twinlen: {}\n\tstride: {}\n').format(
                sublayer['insize'], sublayer['size'], sublayer['winlen'],
                sublayer['stride']))
        layer = Convolution(
            sublayer['insize'], sublayer['size'], sublayer['winlen'],
            stride=sublayer['stride'], fun=tanh)
    elif sublayer['type'] == 'GruMod':
        sys.stderr.write((
            'Loading GRU layer with attributes:\n\tin size: {}\n' +
            '\tout size: {}\n').format(
                sublayer['insize'], sublayer['size']))
        layer = GruMod(sublayer['insize'], sublayer['size'])
    elif sublayer['type'] == 'reverse':
        sublayer = sublayer['sublayers']
        sys.stderr.write((
            'Loading Reverse GRU layer with attributes:\n\tin size: {}\n' +
            '\tout size: {}\n').format(
                sublayer['insize'], sublayer['size']))
        layer = Reverse(GruMod(sublayer['insize'], sublayer['size']))
    elif sublayer['type'] == 'GlobalNormTwoState':
        nbase = nbase_flipflop(sublayer['size'])
        sys.stderr.write((
            'Loading flip-flop layer with attributes:\n\tin size: {}\n' +
            '\tnbases: {}\n').format(sublayer['insize'], nbase))
        layer = GlobalNormFlipFlop(sublayer['insize'], nbase)
    elif sublayer['type'] == 'GlobalNormTwoStateCatMod':
        output_alphabet = sublayer['output_alphabet']
        curr_can_base = 0
        collapse_alphabet = ''
        for can_i_nmod in sublayer['can_nmods']:
            collapse_alphabet += output_alphabet[curr_can_base] * (
                can_i_nmod + 1)
            curr_can_base += can_i_nmod + 1
        alphabet_info  = alphabet.AlphabetInfo(
            output_alphabet, collapse_alphabet,
            sublayer['modified_base_long_names'], do_reorder=False)
        sys.stderr.write((
            'Loading modified bases flip-flop layer with attributes:\n' +
            '\tin size: {}\n\tmod bases: {}\n').format(
                sublayer['insize'], alphabet_info.mod_long_names))
        layer = GlobalNormFlipFlopCatMod(sublayer['insize'], alphabet_info)
    else:
        sys.stderr.write('Encountered invalid layer type ({}).'.format(
            sublayer['type']))
        sys.exit(1)

    layer = set_params(layer, sublayer['params'])

    return layer

def main():
    args = parser.parse_args()

    with open(args.json_model) as fp:
        jsn_model = json.load(fp)

    layer_types = [x['type'] for x in jsn_model['sublayers']]
    if len(set(layer_types).difference(COMPATIBLE_LAYERS)) > 0:
        sys.stderr.write((
            'Incompatible layer type(s) ({}) encountered.\n').format(
                ', '.join(set(layer_types).difference(COMPATIBLE_LAYERS))))
        sys.exit(1)

    network = Serial([
        parse_sublayer(sublayer)
        for sublayer in jsn_model['sublayers']])

    torch.save(network, args.output)

    return

if __name__ == '__main__':
    main()
