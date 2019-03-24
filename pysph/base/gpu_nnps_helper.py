from mako.template import Template
from mako.lookup import TemplateLookup
import os
import sys

from compyle.opencl import get_context, profile_kernel, SimpleKernel


def get_simple_kernel(kernel_name, args, src, wgs, preamble=""):
    ctx = get_context()
    knl = SimpleKernel(
        ctx, args, src, wgs,
        kernel_name, preamble=preamble
    )

    return profile_kernel(knl, kernel_name)


def get_elwise_kernel(kernel_name, args, src, preamble=""):
    ctx = get_context()
    from pyopencl.elementwise import ElementwiseKernel
    knl = ElementwiseKernel(
        ctx, args, src,
        kernel_name, preamble=preamble
    )
    return profile_kernel(knl, kernel_name)


def get_reduce_kernel(kernel_name, args, dtype, map_expr,
                    reduce_expr='a+b', neutral='0', preamble=""):
    ctx = get_context()
    from pyopencl.reduction import ReductionKernel
    knl = ReductionKernel(
        ctx, dtype, arguments=args, map_expr=map_expr,
        reduce_expr=reduce_expr, neutral=neutral,
        preamble=preamble
    )
    return profile_kernel(knl, kernel_name)


def get_scan_kernel(kernel_name, args, dtype, input_expr, output_statement,
                    scan_expr='a+b', neutral='0', preamble=""):
    ctx = get_context()
    from pyopencl.algorithm import GenericScanKernel
    knl = GenericScanKernel(
        ctx, dtype, arguments=args, input_expr=input_expr,
        scan_expr=scan_expr, neutral=neutral,
        output_statement=output_statement, preamble=preamble
    )
    return profile_kernel(knl, kernel_name)


class GPUNNPSHelper(object):
    def __init__(self, tpl_filename, backend=None, use_double=False,
                 c_type=None):
        """

        Parameters
        ----------
        tpl_filename
            filename of source template
        backend
            backend to use for helper
        use_double:
            Use double precision floating point data types
        c_type:
            c_type to use. Overrides use_double
        """
        disable_unicode = False if sys.version_info.major > 2 else True

        self.src_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                tpl_filename),
            disable_unicode=disable_unicode,
        )
        self.data_t = "double" if use_double else "float"

        if c_type is not None:
            self.data_t = c_type

        helper_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "gpu_helper_functions.mako"),
            disable_unicode=disable_unicode
        )

        helper_preamble = helper_tpl.get_def("get_helpers").render(
            data_t=self.data_t
        )
        preamble = self.src_tpl.get_def("preamble").render(
            data_t=self.data_t
        )
        self.preamble = "\n".join([helper_preamble, preamble])
        self.cache = {}
        self.backend = backend

    def _get_code(self, kernel_name, **kwargs):
        arguments = self.src_tpl.get_def("%s_args" % kernel_name).render(
            data_t=self.data_t, **kwargs
            )

        src = self.src_tpl.get_def("%s_src" % kernel_name).render(
            data_t=self.data_t, **kwargs
            )

        return arguments, src

    def _get_reduce_code(self, kernel_name, **kwargs):
        arguments = self.src_tpl.get_def("%s_args" % kernel_name).render(
            data_t=self.data_t, **kwargs
            )

        map_expr = self.src_tpl.get_def("map_%s_src" % kernel_name).render(
            data_t=self.data_t, **kwargs
            )

        return arguments, map_expr

    def _get_scan_code(self, kernel_name, **kwargs):
        arguments = self.src_tpl.get_def("%s_args" % kernel_name).render(
            data_t=self.data_t, **kwargs
            )

        inp_expr = self.src_tpl.get_def("inp_%s_src" % kernel_name).render(
            data_t=self.data_t, **kwargs
            )

        out_expr = self.src_tpl.get_def("out_%s_src" % kernel_name).render(
            data_t=self.data_t, **kwargs
            )

        return arguments, inp_expr, out_expr

    def get_kernel(self, kernel_name, **kwargs):
        key = kernel_name, tuple(kwargs.items())
        wgs = kwargs.get('wgs', None)

        if key in self.cache:
            return self.cache[key]
        else:
            args, src = self._get_code(kernel_name, **kwargs)

            if wgs is None:
                knl = get_elwise_kernel(kernel_name, args, src,
                                        preamble=self.preamble)
            else:
                knl = get_simple_kernel(kernel_name, args, src, wgs,
                                        preamble=self.preamble)

            self.cache[key] = knl
            return knl

    def get_reduce_kernel(self, kernel_name, dtype, **kwargs):
        key = kernel_name, dtype, tuple(kwargs.items())
        reduce_expr = kwargs.get('reduce_expr', 'a+b')
        neutral = kwargs.get('neutral', '0')

        if key in self.cache:
            return self.cache[key]
        else:
            args, map_expr = self._get_reduce_code(
                    kernel_name, **kwargs
                    )

            knl = get_reduce_kernel(kernel_name, args, dtype, map_expr,
                                    reduce_expr=reduce_expr,
                                    neutral=neutral, preamble=self.preamble)

            self.cache[key] = knl
            return knl

    def get_scan_kernel(self, kernel_name, dtype, **kwargs):
        key = kernel_name, dtype, tuple(kwargs.items())
        scan_expr = kwargs.get('scan_expr', 'a+b')
        neutral = kwargs.get('neutral', '0')

        if key in self.cache:
            return self.cache[key]
        else:
            args, input_expr, output_statement = self._get_scan_code(
                    kernel_name, **kwargs
                    )

            knl = get_scan_kernel(kernel_name, args, dtype, input_expr,
                                  output_statement, scan_expr=scan_expr,
                                  neutral=neutral, preamble=self.preamble)

            self.cache[key] = knl
            return knl

