import argparse
from collections import namedtuple
import multiprocessing
import numpy as np
import os
import re
import warnings

"""ArgParse extensions.

Contains many actions for parsing arguments into explicit types and
checking of values are within explicit sets.

"""


class ByteString(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values.encode('ascii'))


def checkProbabilities(probabilities):
    try:
        for probability in iter(probabilities):
            assert 0.0 <= probability <= 1.0, 'Probability {} not in [0,1]'.format(
                probability)
    except TypeError:
        assert 0.0 <= probabilities <= 1.0, 'Probability not in [0,1]'


class display_version_and_exit(argparse.Action):
    """Ronseal."""

    def __init__(self, **kwdargs):
        self.__version__ = kwdargs['metavar']
        super(display_version_and_exit, self).__init__(**kwdargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(self.__version__)
        exit(0)


class FileExists(argparse.Action):
    """Check if the input file exist."""

    def __call__(self, parser, namespace, values, option_string=None):
        if not os.path.exists(values):
            raise RuntimeError(
                "File/path for '{}' does not exist, {}".format(self.dest, values))
        setattr(namespace, self.dest, values)


class FileExist(FileExists):

    def __init__(self, **kwdargs):
        warnings.warn(
            "FileExist is deprecated. Use FileExists instead.", DeprecationWarning)
        super(FileExist, self).__init__(**kwdargs)


class FileAbsent(argparse.Action):
    """Check that input file doesn't exist."""

    def __call__(self, parser, namespace, values, option_string=None):
        if os.path.exists(values):
            raise RuntimeError(
                "File/path for '{}' exists, {}".format(self.dest, values))
        setattr(namespace, self.dest, values)


class CheckCPU(argparse.Action):
    """Make sure people do not overload the machine"""

    def __call__(self, parser, namespace, values, option_string=None):
        num_cpu = multiprocessing.cpu_count()
        if int(values) <= 0 or int(values) > num_cpu:
            raise RuntimeError(
                'Number of jobs can only be in the range of {} and {}'.format(1, num_cpu))
        setattr(namespace, self.dest, values)


class ParseToNamedTuple(argparse.Action):
    """Parse to a namedtuple
    """

    def __init__(self, **kwdargs):
        assert 'metavar' in kwdargs, "Argument 'metavar' must be defined"
        assert 'type' in kwdargs, "Argument 'type' must be defined"
        assert len(kwdargs['metavar']) == kwdargs[
            'nargs'], 'Number of arguments and descriptions inconstistent'
        assert len(kwdargs['type']) == kwdargs[
            'nargs'], 'Number of arguments and types inconstistent'
        self._types = kwdargs['type']
        kwdargs['type'] = str
        self.Values = namedtuple('Values', ' '.join(kwdargs['metavar']))
        super(ParseToNamedTuple, self).__init__(**kwdargs)
        self.default = self.Values(
            *self.default) if self.default is not None else None

    def __call__(self, parser, namespace, values, option_string=None):
        value_dict = self.Values(*[f(v) for f, v in zip(self._types, values)])
        setattr(namespace, self.dest, value_dict)

    @staticmethod
    def value_as_string(value):
        return ' '.join(str(x) for x in value)


class NegBound(argparse.Action):
    """Create a negative list bound suitable for trimming arrays."""

    def __call__(self, parser, namespace, values, option_string=None):
        if values == 0:
            setattr(namespace, self.dest, None)
        else:
            try:
                setattr(namespace, self.dest, -int(values))
            except:
                raise ValueError(
                    'Illegal value for {} ({}), should be castable to int')


class ExpandRanges(argparse.Action):
    """Translate a str like 1,2,3:5,40 to [1,2,3,4,5,40]"""

    def __call__(self, parser, namespace, values, option_string=None):
        elts = []
        for item in values.replace(' ', '').split(','):
            mo = re.search(r'(\d+):(\d+)', item)
            if mo is not None:
                rng = [int(x) for x in mo.groups()]
                elts.extend(list(range(rng[0], rng[1] + 1)))
            else:
                elts.append(int(item))
        setattr(namespace, self.dest, elts)


class ChannelList(ExpandRanges):

    def __init__(self, **kwdargs):
        warnings.warn(
            "ChannelList is deprecated. Use ExpandRanges instead.", DeprecationWarning)
        super(ChannelList, self).__init__(**kwdargs)


class AutoBool(argparse.Action):

    def __init__(self, option_strings, dest, default=None, required=False, help=None):
        """Automagically create --foo / --no-foo argument pairs"""

        if default is None:
            raise ValueError('You must provide a default with AutoBool action')
        if len(option_strings) != 1:
            raise ValueError(
                'Only single argument is allowed with AutoBool action')
        opt = option_strings[0]
        if not opt.startswith('--'):
            raise ValueError('AutoBool arguments must be prefixed with --')

        opt = opt[2:]
        opts = ['--' + opt, '--no-' + opt]
        if default:
            default_opt = opts[0]
        else:
            default_opt = opts[1]
        super(AutoBool, self).__init__(opts, dest, nargs=0, const=None,
                                       default=default, required=required,
                                       help='{} (Default: {})'.format(help, default_opt))

    def __call__(self, parser, namespace, values, option_strings=None):
        if option_strings.startswith('--no-'):
            setattr(namespace, self.dest, False)
        else:
            setattr(namespace, self.dest, True)

    @staticmethod
    def filter_option_strings(strings):
        for s in strings:
            s = s.strip('-')
            if s[:3] != 'no-':
                yield s


class Maybe(object):
    """Create an argparse argument type that accepts either given type or 'None'

    :param mytype: Type function for type to accept, e.g. `int` or `float`
    """

    def __init__(self, mytype):
        self.mytype = mytype

    def __repr__(self):
        return "None or {}".format(self.mytype)

    def __call__(self, y):
        try:
            if y == 'None':
                res = None
            else:
                res = self.mytype(y)
        except:
            raise argparse.ArgumentTypeError(
                'Argument must be {}'.format(self))
        return res


def TypeOrNone(mytype):
    warnings.warn("TypeOrNone is deprecated. Use Maybe instead.",
                  DeprecationWarning)
    return Maybe(mytype)


class Bounded(object):
    """Create an argparse argument type that accepts values in [lower, upper]

    :param mytype: Type function for type to accept, e.g. `int` or `float`
    """

    def __init__(self, mytype, lower=None, upper=None):
        self.mytype = mytype

        assert lower is not None or upper is not None

        if lower is not None and upper is not None:
            assert lower <= upper

        self.lower = lower
        self.upper = upper

    def __repr__(self):
        if self.lower is not None and self.upper is not None:
            return "{} in range [{}, {}]".format(self.mytype, self.lower, self.upper)
        else:
            if self.lower is not None:
                return "{} in range [{}, inf]".format(self.mytype, self.lower)
            else:
                assert self.upper is not None
                return "{} in range [-inf, {}]".format(self.mytype, self.upper)

    def __call__(self, y):
        yt = self.mytype(y)

        if self.lower is not None and yt < self.lower:
            raise argparse.ArgumentTypeError(
                'Argument must be {}'.format(self))

        if self.upper is not None and yt > self.upper:
            raise argparse.ArgumentTypeError(
                'Argument must be {}'.format(self))

        return yt


def NonNegative(mytype):
    """Create an argparse argument type that accepts only non-negative values

    :param mytype: Type function for type to accept, e.g. `int` or `float`
    """
    return Bounded(mytype, lower=mytype(0))


class Positive(object):
    """Create an argparse argument type that accepts only positive values

    :param mytype: Type function for type to accept, e.g. `int` or `float`
    """

    def __init__(self, mytype):
        self.mytype = mytype

    def __repr__(self):
        return "positive {}".format(self.mytype)

    def __call__(self, y):
        yt = self.mytype(y)
        if yt <= 0:
            raise argparse.ArgumentTypeError(
                'Argument must be {}'.format(self))
        return yt


def proportion(p):
    """Type function for proportion"""
    return Bounded(float, 0.0, 1.0)(p)


def probability(p):
    warnings.warn(
        "probability is deprecated. Use proportion instead.", DeprecationWarning)
    return proportion(p)


def Vector(mytype):
    """Return an argparse.Action that will convert a list of values into a numpy
    array of given type
    """

    class MyNumpyAction(argparse.Action):
        """Parse a list of values into numpy array"""

        def __call__(self, parser, namespace, values, option_string=None):
            try:
                setattr(namespace, self.dest, np.array(values, dtype=mytype))
            except:
                raise argparse.ArgumentTypeError(
                    'Cannot convert {} to array of {}'.format(values, mytype))

        @staticmethod
        def value_as_string(value):
            return ' '.join(str(x) for x in value)
    return MyNumpyAction


def str_to_numeric(x):
    """Up-type a str to either int or float, or leave alone."""
    if not isinstance(x, str):
        return x
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x


class DeviceAction(argparse.Action):
    """Parse string specifying a device (either CPU or GPU) and return a normalised version

    Converts None to 'cpu'
    Converts a string like '2' to int 2
    Converts a string like 'cuda2' to int 2 (for UGE compatibility)
    All other inputs are left as they are
    """

    def __call__(self, parser, namespace, value, option_string=None):
        setattr(namespace, self.dest, self._convert(value))

    def _convert(self, value):
        if value is None:
            return 'cpu'

        # if value is (a string representation of) a positive integer, convert
        # to int
        int_match = re.match('[0-9]+', value)
        if int_match:
            return int(int_match.group())

        # for UGE: convert string of form 'cudaN' to int N
        uge_match = re.match('cuda(?P<id>[0-9]+)', value)
        if uge_match:
            return int(uge_match.group('id'))

        # in all other cases, do nothing, and let torch.device decide
        return value


str_to_type = {
    'None': None,
    'True': True, 'False': False,
    'true': True, 'false': False,
    'TRUE': True, 'FALSE': False
}

bool_actions = {
    AutoBool,
    argparse._StoreTrueAction,
    argparse._StoreFalseAction
}
