#!/usr/bin/env python3
import argparse
import json

from taiyaki.cmdargs import FileExists
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.helpers import file_md5, load_model, open_file_or_stdout
from taiyaki.json import JsonEncoder


def get_parser():
    parser = argparse.ArgumentParser(
        description='Dump JSON representation of model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_common_command_args(parser, ["output"])

    parser.add_argument('model', action=FileExists, help='Model checkpoint')

    return parser


def main():
    args = get_parser().parse_args()
    model_md5 = file_md5(args.model)
    model = load_model(args.model)

    json_out = model.json()
    json_out['md5sum'] = model_md5

    with open_file_or_stdout(args.output) as fh:
        json.dump(json_out, fh, indent=4, cls=JsonEncoder)


if __name__ == "__main__":
    main()
