"""Common OpenCL related functionality.
"""

from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.array  # noqa: 401
import pyopencl.algorithm
import pyopencl.tools
from pyopencl.scan import GenericScanKernel
from pyopencl.elementwise import ElementwiseKernel
from collections import defaultdict
from operator import itemgetter
from mako.template import Template

import logging
logger = logging.getLogger()

from .config import get_config


_ctx = None
_queue = None
_profile_info = defaultdict(float)


def get_context():
    global _ctx
    if _ctx is None:
        _ctx = cl.create_some_context()
    return _ctx


def set_context(ctx):
    global _ctx
    _ctx = ctx


def get_queue():
    global _queue
    if _queue is None:
        properties = None
        if get_config().profile:
            properties = cl.command_queue_properties.PROFILING_ENABLE
        _queue = cl.CommandQueue(get_context(), properties=properties)
    return _queue


def set_queue(q):
    global _queue
    _queue = q


def profile_kernel(kernel, name):
    def _profile_knl(*args):
        event = kernel(*args)
        print(name, event)
        profile(name, event)
    if get_config().profile:
        return _profile_knl
    else:
        return kernel


def profile(name, event):
    global _profile_info
    event.wait()
    time = (event.profile.end - event.profile.start) * 1e-9
    _profile_info[name] += time


def print_profile():
    global _profile_info
    _profile_info = sorted(_profile_info.items(), key=itemgetter(1),
                           reverse=True)
    if len(_profile_info) == 0:
        print("No profile information available")
        return
    print("{:<30} {:<30}".format('Kernel', 'Time'))
    tot_time = 0
    for kernel, time in _profile_info:
        print("{:<30} {:<30}".format(kernel, time))
        tot_time += time
    print("Total profiled time: %g secs" % tot_time)



