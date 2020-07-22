"""
Mostly shamelessly borrowed from:
https://docs.python.org/2/library/itertools.html#recipes

because its all so useful!

"""
from itertools import *
from functools import partial
from multiprocessing import Pool
import sys
import traceback


def empty_iterator(it):
    """ Check if an iterator is empty and prepare a fresh one for use

    Args:
        it (Iterable): Iterator to test

    Returns:
        Tuple of 1) boolean indicating if the iterator is empty and,
            2) iterator as passed in
    """
    it, any_check = tee(it)
    try:
        next(any_check)
    except StopIteration:
        return True, it
    else:
        return False, it


class __NotGiven(object):

    def __init__(self):
        """ Some horrible voodoo """
        pass


def try_except_pass(func, *args, **kwargs):
    """ Try function: if error occurs, print traceback and return None

    When wrapping a function we would ordinarily form a closure over a (sub)set
    of the inputs. Such closures cannot be pickled however since the wrapper
    name is not importable. We get around this by using functools.partial
    (which is pickleable). The result is that we can decorate a function to
    mask exceptions thrown by it.

    Args:
        func (Callable): Function to try
        *args: Arguments to pass to `func`
        **kwargs: Keyword arguments to pass to `func`

    Returns:
        `None` if func raises an exception, else the function return value.
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        exc_info = sys.exc_info()
        traceback.print_tb(exc_info[2])
        return None


def imap_mp(
        function, args, fix_args=__NotGiven(), fix_kwargs=__NotGiven(),
        pass_exception=False, threads=1, unordered=False, chunksize=1,
        init=None, initargs=()):
    """ Map a function using multiple processes

    Args:
        function (Callable): Function to apply, must be pickalable for
            multiprocess mapping (problems will results if the function is not
            at the top level of scope).
        args (Iterable): iterable of argument values of function to map over
        fix_args (Iterable): arguments to hold fixed
        fix_kwargs (dict): keyword arguments to hold fixed
        pass_exception (bool): ignore exceptions thrown by function?
        threads (int): number of subprocesses
        unordered (bool): use unordered multiprocessing map
        chunksize (int): multiprocessing job chunksize
        init (Callable): function to each thread to call when it is created.
        initargs (iterable): list of arguments for init

    .. note::
        This function is a generator, the caller will need to consume this.
        Not all options of all mapping functions are supported (why have a
        wrapper in such cases?). If there is a compelling need for more
        flexibility, it can be added.

    If fix_args or fix_kwargs are given, these are first used to create a
    partially evaluated version of function.

    The special :class:`__NotGiven` is used here to flag when optional
    arguments are to be used.

    Yields:
        Results from function calls
    """
    my_function = function
    if not isinstance(fix_args, __NotGiven):
        my_function = partial(my_function, *fix_args)
    if not isinstance(fix_kwargs, __NotGiven):
        my_function = partial(my_function, **fix_kwargs)

    if pass_exception:
        my_function = partial(try_except_pass, my_function)

    if threads == 1:
        if init is not None:
            init(*initargs)
        for r in map(my_function, args):
            yield r
    else:
        pool = Pool(threads, init, initargs)
        if unordered:
            mapper = pool.imap_unordered
        else:
            mapper = pool.imap
        for r in mapper(my_function, args, chunksize=chunksize):
            yield r
        pool.close()
        pool.join()
