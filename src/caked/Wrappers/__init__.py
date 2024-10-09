from __future__ import annotations

from functools import wraps


def none_return_none(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if args[0] is None:
            return None
        return func(*args, **kwargs)

    return wrapper
