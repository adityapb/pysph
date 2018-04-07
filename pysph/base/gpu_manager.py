# Returns appropriate objects based on OpenCL or cuda()
from pysph.base.config import get_config


def opencl():
    return get_config().use_opencl


def cuda():
    return get_config().use_cuda


if opencl():
    from pysph.base.opencl import get_queue, get_context, profile_kernel
    import pyopencl as cl
    import pyopencl.array as gpu_array
    import pyopencl.algorithm as algorithm
    from pyopencl.scan import GenericScanKernel
    from pyopencl.elementwise import ElementwiseKernel
    from pyopencl.cltypes import make_float3
    from pyopencl.cltypes import make_double3
    from pyopencl.tools import dtype_to_ctype
    from pyopencl._cluda import CLUDA_PREAMBLE

if cuda():
    import pycuda as cu
    import pycuda.gpuarray as gpu_array
    import pycuda.scan as scan
    from pycuda.elementwise import ElementwiseKernel
    #from pycuda.gpuarray.vec import make_float3
    #from pycuda.gpuarray.vec import make_double3
    from pycuda.tools import dtype_to_ctype
    import pycuda.autoinit
    from pycuda._cluda import CLUDA_PREAMBLE


def to_device(array):
    if opencl():
        return gpu_array.to_device(get_queue(), array)
    if cuda():
        return cu.gpuarray.to_gpu(array)


def ones_like(array):
    if opencl():
        return 1 + gpu_array.zeros_like(array)
    if cuda():
        return gpu_array.ones_like(array)


def ones(n, dtype):
    if opencl():
        return 1 + gpu_array.zeros(get_queue(), n, dtype)
    if cuda():
        return 1 + gpu_array.zeros(n, dtype)


def empty(n, dtype):
    if opencl():
        return gpu_array.empty(get_queue(), n, dtype)
    if cuda():
        return gpu_array.empty(n, dtype)


def zeros(n, dtype):
    if opencl():
        return gpu_array.zeros(get_queue(), n, dtype)
    if cuda():
        return gpu_array.zeros(n, dtype)


def arange(start, stop, step, dtype=None):
    if opencl():
        return gpu_array.arange(get_queue(), start, stop, step, dtype=dtype)
    if cuda():
        return gpu_array.arange(start, stop, step, dtype=dtype)


def get_exclusive_scan_kernel(*args, **kwargs):
    if 'name' in kwargs:
        name = kwargs.pop('name')
    if opencl():
        knl = algorithm.ExclusiveScanKernel(get_context(),
                *args, **kwargs)
        return profile_kernel(knl, name)
    if cuda():
        return scan.ExclusiveScanKernel(*args, **kwargs)


def get_generic_scan_kernel(*args, **kwargs):
    if 'name' in kwargs:
        name = kwargs.pop('name')
    if opencl():
        knl = GenericScanKernel(get_context(),
                *args, **kwargs)
        return profile_kernel(knl, name)
    if cuda():
        return GenericScanKernel(*args, **kwargs)


def get_elwise_kernel(kernel_name, args, src, preamble=""):
    if opencl():
        ctx = get_context()
        knl = ElementwiseKernel(
            ctx, args, src,
            kernel_name, preamble=preamble
        )
        return profile_kernel(knl, kernel_name)
    if cuda():
        knl = ElementwiseKernel(
            args, src,
            kernel_name, preamble=preamble
        )
        return knl


def get_radix_sort_kernel(args, key_expr, sort_arg_names,
        scan_kernel=None):
    if scan_kernel is None:
        scan_kernel = GenericScanKernel
    ctx = get_context()
    radix_sort = cl.algorithm.RadixSort(
        ctx, args,
        scan_kernel=scan_kernel, key_expr=key_expr,
        sort_arg_names=sort_arg_names
        )

    return radix_sort


