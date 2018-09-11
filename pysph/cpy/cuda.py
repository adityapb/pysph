"""Common CUDA related functionality.
"""
from __future__ import print_function
from collections import defaultdict
from operator import itemgetter
import numpy as np
from mako.template import Template

from .config import get_config
from .types import dtype_to_ctype

import pycuda as cu
from pycuda.scan import InclusiveScanKernel
from pycuda.elementwise import ElementwiseKernel

_cuda_ctx = False


def set_context():
    global _cuda_ctx
    if not _cuda_ctx:
        import pycuda.autoinit
        _cuda_ctx = True


