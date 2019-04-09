#!/usr/bin/env python3
import argparse
import json

from taiyaki.cmdargs import AutoBool, FileExists, FileAbsent
from taiyaki.helpers import load_model
from taiyaki.json import JsonEncoder

parser = argparse.ArgumentParser(description='Dump JSON representation of model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--out_file', default=None, action=FileAbsent, help='Output JSON file to this file location')
parser.add_argument('--params', default=True, action=AutoBool, help='Output parameters as well as model structure')

parser.add_argument('model', action=FileExists, help='Model file to read from')


def main():
    args = parser.parse_args()
    model = load_model(args.model)

    json_out = model.json(args.params)

    if args.out_file is not None:
        with open(args.out_file, 'w') as f:
            print("Writing to file: ", args.out_file)
            json.dump(json_out, f, indent=4, cls=JsonEncoder)
    else:
        print(json.dumps(json_out, indent=4, cls=JsonEncoder))


if __name__ == "__main__":
    main()
