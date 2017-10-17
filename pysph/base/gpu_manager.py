# Returns appropriate objects based on OpenCL or CUDA
from pysph.base.config import get_config


OPENCL = get_config().use_opencl
CUDA = get_config().use_cuda

if OPENCL:
    from pysph.base.opencl import get_queue, get_context
    import pyopencl as backend
    import pyopencl.array as gpu_array
    import pyopencl.algorithm as algorithm
    from pyopencl.array.vec import make_float3
    from pyopencl.array.vec import make_double3
    from pyopencl.array import sum

if CUDA:
    import pycuda as backend
    import pycuda.gpuarray as gpu_array
    import pycuda.scan as algorithm
    from pycuda.gpuarray.vec import make_float3
    from pycuda.gpuarray.vec import make_double3
    from pycuda.gpuarray import sum


def to_device(array):
    if OPENCL:
        gpu_array.to_device(get_queue(), array)
    if CUDA:
        gpu_array.to_device(array)

def empty(n, dtype):
    if OPENCL:
        return gpu_array.empty(get_queue(), n, dtype)
    if CUDA:
        return gpu_array.empty(n, dtype)

def get_exclusive_scan_kernel(*args, **kwargs):
    if OPENCL:
        return algorithm.ExclusiveScanKernel(get_context(),
                *args, **kwargs)
    if CUDA:
        return algorithm.ExclusiveScanKernel(*args, **kwargs)

def get_exclusive_scan_kernel(*args, **kwargs):
    if OPENCL:
        return algorithm.ExclusiveScanKernel(get_context(),
                *args, **kwargs)
    if CUDA:
        return algorithm.ExclusiveScanKernel(*args, **kwargs)

