"""
Mostly shamelessly borrowed from:
https://docs.python.org/2/library/itertools.html#recipes

because its all so useful!

"""
from collections import deque
from itertools import *
from functools import partial
import operator
import numpy as np
import random
from multiprocessing import Pool
import sys
import traceback


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
    except:
        exc_info = sys.exc_info()
        traceback.print_tb(exc_info[2])
        return None


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


def take(n, iterable):
    """ Return first n items of the iterable as a list

    Args:
        n (int): Number of items to return
        iterable (Iterable): An iterable object

    Returns:
        List containing `n` next items from `iterable`
    """
    return list(islice(iterable, n))


def tabulate(function, start=0):
    """ Apply function to consecutive integer and return iterable.

    Args:
        function (Callable): Function to apply
        start (int): First integer on which to apply `function`

    Returns:
        Iterable with results from applying function to consecutive integers
    """
    return map(function, count(start))


def consume(iterator, n):
    """ Advance the iterator n-steps ahead. If n is none, consume entirely.

    Args:
        iterator (Iterable): An iterable object
        n (int): Number of items to consume
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


def nth(iterable, n, default=None):
    """ Extract nth item or a default value

    Args:
        iterator (Iterable): An iterable object
        n (int): Number of items to consume
        default: Default value when iterator is exhuasted.

    Returns:
        Nth item from iterable or default value
    """
    return next(islice(iterable, n, None), default)


def quantify(iterable, pred=bool):
    """ Count how many times the predicate is true

    Args:
        iterator (Iterable): An iterable object
        pred (Callable): Function to apply to items in iterable

    Returns:
        Interger sum of function return values.
    """
    return sum(map(pred, iterable))


def padnone(iterable):
    """ Returns the sequence elements and then returns None indefinitely.

    Useful for emulating the behavior of the built-in map() function.

    Args:
        iterator (Iterable): An iterable object

    Returns:
       Iterable consisting of iterable followed by None indefinitely
    """
    return chain(iterable, repeat(None))


def ncycles(iterable, n):
    """ Returns the sequence elements n times

    Args:
        iterator (Iterable): An iterable object
        n (int): Number of times to cycle through elements

    Returns:
        Iterable consisting of `n` cycles through `iterable`
    """
    return chain.from_iterable(repeat(tuple(iterable), n))


def dotproduct(vec1, vec2):
    """ Dot product of two vectors

    Args:
        vec1 (Iterable): First input vector
        vec2 (Iterable): Second input vector

    Returns:
        Dot product of two vectors (numeric).
    """
    return sum(map(operator.mul, vec1, vec2))


def flatten(listOfLists):
    """ Flatten one level of nesting

    Args:
        listOfLists (list): List of iterables

    Returns:
        Iterable consisting of items removing first level of nesting
    """
    return chain.from_iterable(listOfLists)


def repeatfunc(func, times=None, *args):
    """ Repeat calls to func with specified arguments `times` times.

    Example:
        repeatfunc(random.random)

    Args:
        func (Callable): Function to apply
        times (int): Number of times to repeat `func` call.
        *args: Arguments to pass to `func`

    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def pairwise(iterable):
    """ s -> (s0, s1), (s1, s2), (s2, s3), ...

    Args:
        iterable (Iterable): An iterable object

    Returns:
        Iterable over pairs of input `iterable` offset by one value.
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable, n, fillvalue=None):
    """ Collect data into fixed-length chunks or blocks

    Args:
        iteratable (Iterable): An iterable object
        n (int): Number of items to consume
        fillvalue: Fill last group with `fillvalue`

    Returns:
        Iterable with input grouped into `n` sized groups
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def grouper_it(iterable, n):
    """ As grouper but doesn't pad final chunk

    Args:
        iteratable (Iterable): An iterable object
        n (int): Number of items to consume

    Returns:
        Iterable with input grouped into `n` sized groups (except possibly
            the last group)
    """
    it = iter(iterable)
    while True:
        chunk_it = islice(it, n)
        try:
            first_el = next(chunk_it)
        except StopIteration:
            return
        yield chain((first_el,), chunk_it)


def blocker(iterable, n):
    """ Yield successive n-sized blocks from iterable as numpy array.
    Doesn't pad final block.

    Args:
        iterator (Iterable): An iterable object
        n (int): Number of items to consume

    Yields:
        np.array containing slice of iterable.
    """
    for i in range(0, len(iterable), n):
        yield np.array(iterable[i:i + n])


def roundrobin(*iterables):
    """ roundrobin('ABC', 'D', 'EF') --> A D E B F C

    Args:
        *iterables: Iterables to be consumed

    Yields:
        Element from input iterables
    """
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while pending:
        try:
            for callable_next in nexts:
                yield callable_next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))


def powerset(iterable):
    """ powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    Args:
        iterator (Iterable): An iterable object

    Returns:
        Iterable over sets from input iterable
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(iterable, key=None):
    """ List unique elements, preserving order. Remember all elements
    ever seen.

    Args:
        iterable (Iterable): An iterable object
        key (Callable): Function to extract key from items

    Yields:
        Unique items from `iterable`
    """
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable, key=None):
    """ List unique elements, preserving order. Remember only the element
    just seen.

    Args:
        iterable (Iterable): An iterable object
        key (Callable): Function to extract key from items

    Yields:
        Items from `iterable` not immediately repeated
    """
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return map(next, map(itemgetter(1), groupby(iterable, key)))


def iter_except(func, exception, first=None):
    """ Call a function repeatedly until an exception is raised.

    Converts a call-until-exception interface to an iterator interface.
    Like __builtin__.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.

    Examples:
        bsddbiter = iter_except(db.next, bsddb.error, db.first)
        heapiter = iter_except(functools.partial(heappop, h), IndexError)
        dictiter = iter_except(d.popitem, KeyError)
        dequeiter = iter_except(d.popleft, IndexError)
        queueiter = iter_except(q.get_nowait, Queue.Empty)
        setiter = iter_except(s.pop, KeyError)

    Args:
        func (Callable): Function to call
        exception (Exception): Exception type to catch
        first (Callable): Function to call and yield before func

    Yields:
        Return values from `first` or `func`
    """
    try:
        if first is not None:
            yield first()
        while 1:
            yield func()
    except exception:
        pass


def random_product(*args, **kwds):
    """ Random selection from itertools.product()

    Args:
        *args: Positional arguments
        **kwds: Keywork arguments

    Returns:
        Tuple containing random selection of inputs.
    """
    pools = list(map(tuple, args)) * kwds.get('repeat', 1)
    return tuple(random.choice(pool) for pool in pools)


def random_permutation(iterable, r=None):
    """ Random selection from itertools.permutations(iterable, r)

    Args:
        iterable (Iterable): Iterable object
        r (int): Number of values to select

    Return:
        Tuple of items from iterable
    """
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def random_combination(iterable, r):
    """ Random selection from itertools.combinations(iterable, r)

    Args:
        iterable (Iterable): Iterable object
        r (int): Number of values to select

    Return:
        Tuple of random combination from iterable.
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable, r):
    """ Random selection from
    itertools.combinations_with_replacement(iterable, r)

    Args:
        iterable (Iterable): Iterable object
        r (int): Number of values to select

    Return:
        Tuple of random combination with replacement from iterable.
    """
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.randrange(n) for i in range(r))
    return tuple(pool[i] for i in indices)


def tee_lookahead(t, i):
    """ Inspect the i-th upcomping value from a tee object while leaving the
    tee object at its current position.

    Raises:
        IndexError: If underlying iterator doesn't have enough values.

    Args:
        t (Iterable): Iterable object
        i (int): Number of items to lookahead
    """
    for value in islice(t.__copy__(), i, None):
        return value
    raise IndexError(i)


def window(iterable, size):
    """ Create an iterator returning a sliding window from another iterator

    Args:
        iterable (Iterable): Iterable object
        size (int): Size of window

    Returns:
        Iterator returning a tuple containing the data in the window
    """
    assert size > 0, "Window size for iterator should be strictly positive, got {0}".format(
        size)
    iters = tee(iterable, size)
    for i in range(1, size):
        for each in iters[i:]:
            next(each, None)
    return zip(*iters)


def centered_truncated_window(iterable, size):
    """ A sliding window generator padded with shorter windows at edges,
    output is the same length as the input. Will pad on the right more.
    [1,2,3,4,5] -> (1,2), (1,2,3), (2,3,4), (3,4,5), (4,5)

    ArgS:
        iterable (Iterable): Iterable object
        size (int): Size of window

    Yeilds:
        items from windows
    """
    edge, bulk = tee(iterable, 2)
    edge = take(size + 1, edge)
    for i in range(size // 2 + 1, size):
        yield tuple(edge[:i])

    # bulk can be handled by window()
    count = 0
    for win in window(bulk, size):
        yield win
        count += 1

    edge = list(win)[1:]
    for i in range(size // 2):
        yield tuple(edge[i:])


class __NotGiven(object):

    def __init__(self):
        """ Some horrible voodoo """
        pass


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
