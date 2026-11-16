"""TTY-only progress bars, including the untouched upstream sampling loops."""
import os
import sys
from tqdm.auto import tqdm as _tqdm


class tqdm(_tqdm):
    def __init__(self, *args, **kwargs):
        stream = kwargs.get('file', sys.stderr)
        terminal = os.environ.get('MOMASK_TTY') == '1' or stream.isatty()
        kwargs['disable'] = bool(kwargs.get('disable', False)) or not terminal
        kwargs.setdefault('mininterval', 1.)
        super().__init__(*args, **kwargs)


def install():
    import tqdm.auto
    tqdm.auto.tqdm = globals()['tqdm']
