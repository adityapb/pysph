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


output_elwise_template = '''
${dtype} item, prev_item;
item = scan_out[i];
if(i < 1)
    prev_item = neutral;
else
    prev_item = scan_out[i-1];
${output_statement}
'''


def set_context():
    global _cuda_ctx
    if not _cuda_ctx:
        import pycuda.autoinit
        _cuda_ctx = True


class GenericScanKernel(object):
    def __init__(self, dtype, arguments, input_expr, scan_expr, neutral,
            output_statement, is_segment_start_expr=None,
            input_fetch_exprs=[], index_dtype=np.int32, name_prefix='scan',
            options=[], preamble="", devices=None):
        # Combination of an inclusive scan and 2 elementwise kernels
        self.dtype = dtype
        self.neutral = neutral

        # Input elementwise kernel
        args_inp_elwise = ", ".join([arguments,
            "%s *scan_inp" % dtype_to_ctype(dtype)])
        inp_oper = "scan_inp[i] = %s" % input_expr
        self.inp_elwise_knl = ElementwiseKernel(args_inp_elwise, inp_oper,
                                                name="inp_elwise_knl",
                                                preamble=preamble)

        # Inclusive scan kernel
        self.scan_knl = InclusiveScanKernel(dtype, scan_expr, neutral=neutral,
                                            preamble=preamble)

        # Output elementwise kernel
        args_out_elwise = ", ".join([arguments,
            "%(dtype)s *scan_out, %(dtype)s neutral" % {'dtype':dtype_to_ctype(dtype)}])
        template = Template(text=output_elwise_template)
        out_oper = template.render(
                dtype=dtype_to_ctype(dtype),
                output_statement=output_statement
                )
        self.out_elwise_knl = ElementwiseKernel(args_out_elwise, out_oper,
                                                name="out_elwise_knl",
                                                preamble=preamble)

    def __call__(self, *args):
        scan_inp = cu.gpuarray.empty(args[0].size, dtype=self.dtype)
        inp_elwise_args = args + (scan_inp,)
        # Input elementwise
        self.inp_elwise_knl(*inp_elwise_args)

        # Scan
        scan_out = self.scan_knl(scan_inp)

        neutral = np.asarray(self.neutral, dtype=self.dtype)
        out_elwise_args = args + (scan_out, neutral)
        # Output elementwise
        self.out_elwise_knl(*out_elwise_args)

