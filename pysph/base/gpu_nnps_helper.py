#!/usr/bin/env python

import pyopencl as cl
from pyopencl.elementwise import ElementwiseKernel
from mako.template import Template
import os

from pysph.base.opencl import get_queue, profile_kernel


class GPUNNPSHelper(object):
    def __init__(self, ctx, tpl_filename, use_double=False):
        self.src_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                tpl_filename),
            disable_unicode=True
        )

        self.data_t = "double" if use_double else "float"

        helper_tpl = Template(
            filename=os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                "gpu_helper_functions.mako"),
            disable_unicode=True
        )

        helper_preamble = helper_tpl.get_def("get_helpers").render(
            data_t=self.data_t
        )
        preamble = self.src_tpl.get_def("preamble").render(
            data_t=self.data_t
        )
        self.preamble = "\n".join([helper_preamble, preamble])
        self.ctx = ctx
        self.queue = get_queue()
        self.cache = {}

    def _get_code(self, kernel_name, **kwargs):
        knl = self.src_tpl.get_def(kernel_name).render(
                data_t=self.data_t, **kwargs)

        return "\n".join([self.preamble, knl])

    def get_kernel(self, kernel_name, **kwargs):
        data = kernel_name, tuple(kwargs.items())
        if data in self.cache:
            return profile_kernel(self.cache[data], kernel_name)
        else:
            src = self._get_code(kernel_name, **kwargs)
            prg = cl.Program(self.ctx, src).build()
            def call(*args):
                knl = getattr(prg, kernel_name)
                event = knl(self.queue, args[1].size, None, *args)
                return event
            self.cache[data] = call
            return profile_kernel(call, kernel_name)
