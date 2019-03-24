#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map

from cython.operator cimport dereference as deref, preincrement as inc

# Cython for compiler directives
cimport cython

import pyopencl as cl
import pyopencl.array
import pyopencl.algorithm
from pyopencl.scan import GenericScanKernel
from pyopencl.elementwise import ElementwiseKernel

import numpy as np
cimport numpy as np
from mako.template import Template

from pysph.base.gpu_nnps_helper import GPUNNPSHelper
from compyle.array import Array
import compyle.array as array
from compyle.opencl import get_context, get_config, profile_kernel


IF UNAME_SYSNAME == "Windows":
    cdef inline double fmin(double x, double y) nogil:
        return x if x < y else y
    cdef inline double fmax(double x, double y) nogil:
        return x if x > y else y

    @cython.cdivision(True)
    cdef inline double log2(double n) nogil:
        return log(n)/log(2)


cdef class ZOrderGPUNNPS(GPUNNPS):
    def __init__(self, int dim, list particles, double radius_scale=2.0,
            int ghost_layers=1, domain=None, bint fixed_h=False,
            bint cache=True, bint sort_gids=False, backend='opencl'):
        GPUNNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids, backend
        )

        self.radius_scale2 = radius_scale*radius_scale
        self.radix_sort = None
        self.make_vec = cl.cltypes.make_double3 if self.use_double \
                else cl.cltypes.make_float3

        self.helper = GPUNNPSHelper("z_order_gpu_nnps.mako",
                                    use_double=self.use_double,
                                    backend=self.backend)

        self.src_index = -1
        self.dst_index = -1
        self.sort_gids = sort_gids

        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles
        self.pids = []
        self.pid_keys = []
        self.cids = []
        self.cid_to_idx = []
        self.lengths = []

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper>self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids.append(Array(np.uint32, n=num_particles,
                             backend=self.backend))
            self.pid_keys.append(Array(np.uint64, n=num_particles,
                                 backend=self.backend))
            self.cids.append(Array(np.uint32, n=num_particles,
                             backend=self.backend))
            self.cid_to_idx.append(Array(np.int32,
                                   backend=self.backend))
            self.lengths.append(Array(np.uint32,
                                backend=self.backend))

        self.max_cid_src = array.zeros(1, dtype=np.int32,
                                       backend=self.backend)

        self.dst_to_src = Array(np.uint32, backend=self.backend)
        self.overflow_cid_to_idx = Array(np.int32, backend=self.backend)

        self.domain.update()
        self.update()

    def get_spatially_ordered_indices(self, int pa_index):
        def update_pids():
            pids_new = array.arange(0, num_particles, 1, dtype=np.uint32,
                                    backend=self.backend)
            self.pids[pa_index].set_data(pids_new)

        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        num_particles = pa_wrapper.get_number_of_particles()
        self.sorted = True
        return self.pids[pa_index], update_pids

    cpdef _bin(self, int pa_index):
        self.sorted = False
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]

        fill_pids = self.helper.get_kernel("fill_pids")

        pa_gpu = pa_wrapper.pa.gpu
        fill_pids(pa_gpu.x.dev, pa_gpu.y.dev, pa_gpu.z.dev,
                self.cell_size,
                self.make_vec(self.xmin[0], self.xmin[1], self.xmin[2]),
                self.pid_keys[pa_index].dev, self.pids[pa_index].dev)

        if self.radix_sort is None:
            self.radix_sort = cl.algorithm.RadixSort(
                get_context(),
                "unsigned int* pids, unsigned long* keys",
                scan_kernel=GenericScanKernel, key_expr="keys[i]",
                sort_arg_names=["pids", "keys"]
            )

        cdef double max_length = fmax(fmax((self.xmax[0] - self.xmin[0]),
            (self.xmax[1] - self.xmin[1])), (self.xmax[2] - self.xmin[2]))

        cdef int max_num_cells = (<int> ceil(max_length/self.hmin))

        cdef int max_num_bits = 3*(<int> ceil(log2(max_num_cells)))

        allocator = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))

        (sorted_indices, sorted_keys), evnt = self.radix_sort(
            self.pids[pa_index].dev, self.pid_keys[pa_index].dev,
            key_bits=max_num_bits, allocator=allocator
        )
        self.pids[pa_index].set_data(sorted_indices)
        self.pid_keys[pa_index].set_data(sorted_keys)

        find_num_unique_cids = self.helper.get_reduce_kernel(
                "find_num_unique_cids", np.int32
                )

        cdef unsigned int num_cids = 1 + \
                <unsigned int> find_num_unique_cids(
                        self.pid_keys[pa_index].dev
                        ).get()

        self.lengths[pa_index].resize(num_cids)
        start_cid = array.empty(num_cids, np.uint32, backend=self.backend)

        fill_unique_cids = self.helper.get_scan_kernel("fill_unique_cids",
                                                       np.uint32)

        fill_unique_cids(self.pid_keys[pa_index].dev,
                         self.cids[pa_index].dev,
                         start_cid.dev)

        cdef unsigned int num_particles = pa_gpu.get_number_of_particles()

        fill_length_cids = self.helper.get_kernel("fill_length_cids")

        fill_length_cids(start_cid.dev, self.lengths[pa_index].dev,
                         num_cids, num_particles)

        print self.lengths[pa_index].dev

        cdef int max_np_wg = array.maximum(self.lengths[pa_index])
        # 32 is used be default because different wgs every iteration
        # will require template rendering every iteration 
        self.wgs[pa_index] = max(2 ** (<int> ceil(log2(max_np_wg))), 32)
        # Use wgs dest index

        curr_cids = self.cids[pa_index]
        self.cid_to_idx[pa_index].resize(27 * num_cids)
        self.cid_to_idx[pa_index].fill(-1)

        self.max_cid[pa_index] = num_cids

        map_cid_to_idx = self.helper.get_kernel("map_cid_to_idx")

        map_cid_to_idx(
            pa_gpu.x.dev, pa_gpu.y.dev, pa_gpu.z.dev,
            pa_wrapper.get_number_of_particles(), self.cell_size,
            self.make_vec(self.xmin[0], self.xmin[1], self.xmin[2]),
            self.pids[pa_index].dev, self.pid_keys[pa_index].dev,
            self.cids[pa_index].dev, self.cid_to_idx[pa_index].dev
        )

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles
        self.max_cid = []
        self.wgs = []
        self.sorted = False

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper>self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i].resize(num_particles)
            self.pid_keys[i].resize(num_particles)
            self.cids[i].resize(num_particles)
            self.max_cid.append(0)
            self.wgs.append(0)

    cpdef set_context(self, int src_index, int dst_index):
        """Setup the context before asking for neighbors.  The `dst_index`
        represents the particles for whom the neighbors are to be determined
        from the particle array with index `src_index`.

        Parameters
        ----------

         src_index: int: the source index of the particle array.
         dst_index: int: the destination index of the particle array.
        """
        GPUNNPS.set_context(self, src_index, dst_index)

        self.src = self.pa_wrappers[src_index]
        self.dst = self.pa_wrappers[dst_index]

        self.dst_src = src_index != dst_index

        cdef unsigned int overflow_size = 0

        if self.dst_src:
            self.dst_to_src.resize(self.max_cid[dst_index])

            map_dst_to_src = self.helper.get_kernel("map_dst_to_src")

            self.max_cid_src.fill(self.max_cid[src_index])

            map_dst_to_src(self.dst_to_src.dev, self.cids[dst_index].dev,
                    self.cid_to_idx[dst_index].dev,
                    self.pid_keys[dst_index].dev,
                    self.pid_keys[src_index].dev, self.cids[src_index].dev,
                    self.src.get_number_of_particles(), self.max_cid_src.dev)

            overflow_size = <unsigned int>(self.max_cid_src.get()) - \
                    self.max_cid[src_index]

            self.overflow_cid_to_idx.resize(max(1, 27 * overflow_size))
            self.overflow_cid_to_idx.fill(-1)

            fill_overflow_map = self.helper.get_kernel("fill_overflow_map")

            dst_gpu = self.dst.pa.gpu
            fill_overflow_map(self.dst_to_src.dev,
                    self.cid_to_idx[dst_index].dev, dst_gpu.x.dev, dst_gpu.y.dev,
                    dst_gpu.z.dev, self.src.get_number_of_particles(),
                    self.cell_size,
                    self.make_vec(self.xmin[0], self.xmin[1],
                        self.xmin[2]),
                    self.pid_keys[src_index].dev, self.pids[dst_index].dev,
                    self.overflow_cid_to_idx.dev,
                    <unsigned int> self.max_cid[src_index])


    cdef void find_neighbor_lengths(self, nbr_lengths):
        z_order_nbr_lengths = self.helper.get_kernel(
                "z_order_nbr_lengths", sorted=self.sorted,
                dst_src=self.dst_src, wgs=self.wgs[self.dst_index])

        ls = self.wgs[self.dst_index]
        gs = ls * self.max_cid[self.dst_index]

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu

        np_src = np.asarray(self.src.get_number_of_particles(),
                            dtype=np.uint32)
        max_cid_src = np.asarray(self.max_cid[self.src_index],
                                 dtype=np.uint32)
        max_cid_dst = np.asarray(self.max_cid[self.dst_index],
                                 dtype=np.uint32)
        radius_scale2 = np.asarray(self.radius_scale2,
                                   dtype=self.dtype)
        cell_size = np.asarray(self.cell_size,
                               dtype=self.dtype)

        #print self.cid_to_idx[self.dst_index]
        #print self.pids[self.dst_index]
        #print self.cids[self.dst_index]

        working_qids = Array(np.uint32, n=self.dst.get_number_of_particles(),
                             backend=self.backend)

        z_order_nbr_lengths(dst_gpu.x.dev, dst_gpu.y.dev, dst_gpu.z.dev,
                dst_gpu.h.dev, src_gpu.x.dev, src_gpu.y.dev, src_gpu.z.dev,
                src_gpu.h.dev,
                self.make_vec(self.xmin[0], self.xmin[1],
                    self.xmin[2]), np_src,
                self.pid_keys[self.src_index].dev,
                self.pids[self.dst_index].dev,
                self.pids[self.src_index].dev,
                max_cid_src, max_cid_dst,
                self.cid_to_idx[self.dst_index].dev,
                self.cids[self.dst_index].dev,
                self.cid_to_idx[self.src_index].dev,
                self.overflow_cid_to_idx.dev, self.dst_to_src.dev,
                self.lengths[self.dst_index].dev,
                self.lengths[self.src_index].dev,
                nbr_lengths.dev, radius_scale2, cell_size, working_qids.dev,
                ls=(ls,), gs=(gs,), queue=self.queue)

        print "qids = \n", working_qids
        #print "Neighbor lengths", nbr_lengths.dev

    cdef void find_nearest_neighbors_gpu(self, nbrs, start_indices):
        z_order_nbrs = self.helper.get_kernel(
                "z_order_nbrs", sorted=self.sorted,
                dst_src=self.dst_src, wgs=self.wgs[self.dst_index])

        ls = self.wgs[self.dst_index]
        gs = ls * self.max_cid[self.dst_index]

        dst_gpu = self.dst.pa.gpu
        src_gpu = self.src.pa.gpu

        np_src = np.asarray(self.src.get_number_of_particles(),
                            dtype=np.uint32)
        max_cid_src = np.asarray(self.max_cid[self.src_index],
                                 dtype=np.uint32)
        max_cid_dst = np.asarray(self.max_cid[self.dst_index],
                                 dtype=np.uint32)
        radius_scale2 = np.asarray(self.radius_scale2,
                                   dtype=self.dtype)
        cell_size = np.asarray(self.cell_size,
                               dtype=self.dtype)

        z_order_nbrs(dst_gpu.x.dev, dst_gpu.y.dev, dst_gpu.z.dev,
                dst_gpu.h.dev, src_gpu.x.dev, src_gpu.y.dev, src_gpu.z.dev,
                src_gpu.h.dev,
                self.make_vec(self.xmin[0], self.xmin[1],
                    self.xmin[2]),
                np_src, self.pid_keys[self.src_index].dev,
                self.pids[self.dst_index].dev,
                self.pids[self.src_index].dev,
                max_cid_src, max_cid_dst,
                self.cid_to_idx[self.dst_index].dev,
                self.cids[self.dst_index].dev,
                self.cid_to_idx[self.src_index].dev,
                self.overflow_cid_to_idx.dev, self.dst_to_src.dev,
                self.lengths[self.dst_index].dev,
                self.lengths[self.src_index].dev,
                start_indices.dev, nbrs.dev, radius_scale2, cell_size,
                ls=(ls,), gs=(gs,), queue=self.queue)
