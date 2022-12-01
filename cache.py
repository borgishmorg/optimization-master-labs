import sys
import functools
from numbers import Number
from typing import Callable
from collections import deque
from time import sleep
from datetime import datetime, timedelta
from collections import deque
from collections.abc import Set, Mapping


# Написать декоратор, который кеширует результаты вызова функции.
# В качестве аргументов принимает: время жизни кешируемых значений,
# максимальное количество кешируемых данных, максимальный размер
# в памяти кешируемых данных, нужно ли выдавать подробную информацию
# о работе функции в консоль. В случае, если вызывается функция с
# теми же аргументами, для которых есть закешированный результат,
# то выдается результат из кеша.


ZERO_DEPTH_BASES = (str, bytes, Number, range, bytearray)


def getsize(obj_0):
    """
Recursively iterate to sum size of object & members.

https://stackoverflow.com/a/30316760
"""
    _seen_ids = set()

    def inner(obj):
        obj_id = id(obj)
        if obj_id in _seen_ids:
            return 0
        _seen_ids.add(obj_id)
        size = sys.getsizeof(obj)
        if isinstance(obj, ZERO_DEPTH_BASES):
            pass  # bypass remaining control flow and return
        elif isinstance(obj, (tuple, list, Set, deque)):
            size += sum(inner(i) for i in obj)
        elif isinstance(obj, Mapping) or hasattr(obj, 'items'):
            size += sum(inner(k) + inner(v)
                        for k, v in getattr(obj, 'items')())
        # Check for custom object instances - may subclass above too
        if hasattr(obj, '__dict__'):
            size += inner(vars(obj))
        if hasattr(obj, '__slots__'):  # can have __slots__ with __dict__
            size += sum(inner(getattr(obj, s))
                        for s in obj.__slots__ if hasattr(obj, s))
        return size
    return inner(obj_0)


def cache(
    ttl: timedelta = None,
    max_size: int = None,
    max_mem_size: int = None,
    verbose: bool = False,
):
    def _decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            # Delete expired keys
            if ttl:
                while (
                    _wrapper._data_ttl_q
                    and _wrapper._data_ttl_q[0][1] < datetime.now()
                ):
                    key, _ = _wrapper._data_ttl_q.popleft()
                    if key in _wrapper._data:
                        del _wrapper._data[key]
                        if verbose:
                            print(
                                f'Removed value for {key=} from {f.__name__} cache'
                            )
                _wrapper._data_mem_size = getsize(_wrapper._data)

            # Calculate key
            key = (args, tuple(kwargs.items()))

            # Return value from cache if exists
            if key in _wrapper._data:
                if verbose:
                    print(f'Returned value for {key=} from {f.__name__} cache')
                return _wrapper._data[key]

            # Calculate actual value
            value = f(*args, **kwargs)

            # Save value in cache
            if (
                (
                    max_size is None
                    or len(_wrapper._data) < max_size
                ) and (
                    max_mem_size is None
                    or _wrapper._data_mem_size + getsize((key, value)) < max_mem_size
                )
            ):
                _wrapper._data[key] = value
                _wrapper._data_mem_size = getsize(_wrapper._data)
                if ttl:
                    _wrapper._data_ttl_q.append((key, datetime.now() + ttl))
                if verbose:
                    print(f'Store value for {key=} in {f.__name__} cache')
            elif (
                verbose
                and max_size is not None
                and len(_wrapper._data) >= max_size
            ):
                print(
                    f'Can\'t store value for {key=} in {f.__name__} cache (cache size overflow)'
                )
            elif (
                verbose
                and max_mem_size is not None
                and _wrapper._data_mem_size + getsize((key, value)) >= max_mem_size
            ):
                print(
                    f'Can\'t store value for {key=} in {f.__name__} cache (cache memory limit overflow)'
                )

            # Return value
            return value

        _wrapper._data = dict()
        _wrapper._data_mem_size = getsize(_wrapper._data)
        if ttl:
            _wrapper._data_ttl_q = deque()

        return _wrapper
    return _decorator


@cache(
    max_size=50,
    # max_mem_size=10000,
    max_mem_size=20000,
    ttl=timedelta(seconds=1),
    # verbose=True,
)
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n-2) + fib(n-1)


def print_info(f):
    print('Data size:', len(getattr(f, '_data', [])))
    print('Data mem size:', getattr(f, '_data_mem_size', None))
    if data_ttl_q := getattr(f, '_data_ttl_q', None):
        match len(data_ttl_q):
            case 0:
                print('TTL queue:', [])
            case 1:
                print('TTL queue:', [data_ttl_q[0]])
            case 2:
                print('TTL queue:', [data_ttl_q[0], data_ttl_q[-1]])
            case _:
                print('TTL queue:', [data_ttl_q[0], '...', data_ttl_q[-1]])
        print('TTL queue len:', len(data_ttl_q))
    else:
        print('TTL queue:', None)


if __name__ == '__main__':
    print(fib(25))
    print_info(fib)
    print()

    print(fib(52))
    print_info(fib)
    print()

    sleep(1)
    print(fib(25))
    print_info(fib)
    print()
