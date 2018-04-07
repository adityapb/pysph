# encoding=utf-8
import pycuda
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np

from cgen import *
from codepy.bpl import BoostPythonModule
from codepy.cuda import CudaModule
from pycuda.tools import dtype_to_ctype

import sys

reload(sys)
sys.setdefaultencoding('utf8')


def get_sort_mod(dtype_map):
    #Make a host_module, compiled for CPU
    host_mod = BoostPythonModule()

    #Make a device module, compiled with NVCC
    nvcc_mod = CudaModule(host_mod)

    #Describe device module code
    #NVCC includes
    nvcc_includes = [
        "thrust/sort.h",
        "thrust/device_vector.h",
        "cuda.h",
        ]
    #Add includes to module
    nvcc_mod.add_to_preamble([Include(x) for x in nvcc_includes])

    #NVCC function
    nvcc_function = FunctionBody(
        FunctionDeclaration(Value("void", "sort_wrapper"),
                            [Value("CUdeviceptr", "input_ptr"),
                             Value("int", "length")]),
        Block([Statement("thrust::device_ptr<%(value)s> thrust_ptr((%(value)s*)input_ptr)" % dtype_map),
            Statement("thrust::sort(thrust_ptr, thrust_ptr+length)")]))

    #Add declaration to nvcc_mod
    #Adds declaration to host_mod as well
    nvcc_mod.add_function(nvcc_function)

    host_includes = [
        "boost/python/extract.hpp",
        ]
    #Add host includes to module
    host_mod.add_to_preamble([Include(x) for x in host_includes])

    host_namespaces = [
        "using namespace boost::python",
        ]

    #Add BPL using statement
    host_mod.add_to_preamble([Statement(x) for x in host_namespaces])


    host_statements = [
        #Extract information from PyCUDA GPUArray
        #Get length
        "tuple shape = extract<tuple>(gpu_array.attr(\"shape\"))",
        "int length = extract<int>(shape[0])",
        #Get data pointer
        "CUdeviceptr input_ptr = extract<CUdeviceptr>(gpu_array.attr(\"ptr\"))",
        #Call Thrust routine, compiled into the CudaModule
        "sort_wrapper(input_ptr, length)",
        #Return result
        "return gpu_array",
        ]

    host_mod.add_function(
        FunctionBody(
            FunctionDeclaration(Value("object", "host_entry"),
                                [Value("object", "gpu_array")]),
            Block([Statement(x) for x in host_statements])))


    #Compile modules
    import codepy.jit, codepy.toolchain
    gcc_toolchain = codepy.toolchain.guess_toolchain()
    nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()

    module = nvcc_mod.compile(gcc_toolchain, nvcc_toolchain, debug=True)
    return module



def get_sort_by_key_mod(dtype_map):

    #Make a host_module, compiled for CPU
    host_mod = BoostPythonModule()

    #Make a device module, compiled with NVCC
    nvcc_mod = CudaModule(host_mod)

    #Describe device module code
    #NVCC includes
    nvcc_includes = [
        "thrust/sort.h",
        "thrust/device_vector.h",
        "thrust/execution_policy.h",
        "cuda.h",
        ]
    #Add includes to module
    nvcc_mod.add_to_preamble([Include(x) for x in nvcc_includes])

    #NVCC function
    nvcc_function = FunctionBody(
        FunctionDeclaration(Value("void", "sort_by_key_wrapper"),
                            [Value("CUdeviceptr", "input_ptr"),
                             Value("CUdeviceptr", "key_ptr"),
                             Value("int", "length")]),
        Block([Statement("thrust::device_ptr<%(value)s> thrust_ptr((%(value)s*)input_ptr)" % dtype_map),
            Statement("thrust::device_ptr<%(key)s> key_thrust_ptr((%(key)s*)key_ptr)" % dtype_map),
            Statement("thrust::sort_by_key(key_thrust_ptr, key_thrust_ptr+length, thrust_ptr)")]))

    #Add declaration to nvcc_mod
    #Adds declaration to host_mod as well
    nvcc_mod.add_function(nvcc_function)

    host_includes = [
        "boost/python/extract.hpp",
        ]
    #Add host includes to module
    host_mod.add_to_preamble([Include(x) for x in host_includes])

    host_namespaces = [
        "using namespace boost::python",
        ]

    #Add BPL using statement
    host_mod.add_to_preamble([Statement(x) for x in host_namespaces])


    host_statements = [
        #Extract information from PyCUDA GPUArray
        #Get length
        "tuple shape = extract<tuple>(gpu_array.attr(\"shape\"))",
        "int length = extract<int>(shape[0])",
        #Get data pointer
        "CUdeviceptr input_ptr = extract<CUdeviceptr>(gpu_array.attr(\"ptr\"))",
        "CUdeviceptr key_ptr = extract<CUdeviceptr>(key_array.attr(\"ptr\"))",
        #Call Thrust routine, compiled into the CudaModule
        "sort_by_key_wrapper(input_ptr, key_ptr, length)",
        #Return result
        "return gpu_array",
        ]

    host_mod.add_function(
        FunctionBody(
            FunctionDeclaration(Value("object", "host_entry"),
                                [Value("object", "gpu_array"), Value("object", "key_array")]),
            Block([Statement(x) for x in host_statements])))


    #Compile modules
    import codepy.jit, codepy.toolchain
    gcc_toolchain = codepy.toolchain.guess_toolchain()
    nvcc_toolchain = codepy.toolchain.guess_nvcc_toolchain()

    module = nvcc_mod.compile(gcc_toolchain, nvcc_toolchain, debug=True)
    return module


def sort(values):
    dtype_map = {'value': dtype_to_ctype(values.dtype)}
    mod = get_sort_mod(dtype_map)
    return mod.host_entry(values)


def sort_by_key(values, keys):
    dtype_map = {'value': dtype_to_ctype(values.dtype),
                 'key' : dtype_to_ctype(keys.dtype)}
    mod = get_sort_by_key_mod(dtype_map)
    return mod.host_entry(values, keys)


if __name__ == "__main__":
    length = 100
    a = np.array(np.random.rand(length), dtype=np.float32)
    x = np.arange(0, length, 1, dtype=np.float32)
    print("---------------------- Unsorted -----------------------")
    print(x)
    b = gpuarray.to_gpu(a)
    y = gpuarray.to_gpu(x)
    # Call Thrust!!
    c = sort_by_key(y, b)
    print("----------------------- Sorted ------------------------")
    print(b.get())
    print("-------------------------------------------------------")
