#cython: embedsignature=True

# malloc and friends
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.map cimport map

from cython.operator cimport dereference as deref, preincrement as inc

# Cython for compiler directives
cimport cython

import numpy as np
cimport numpy as np

cdef extern from "<algorithm>" namespace "std" nogil:
    void sort[Iter, Compare](Iter first, Iter last, Compare comp)
    void sort[Iter](Iter first, Iter last)

#############################################################################

cdef class ZOrderNNPS(NNPS):

    """Find nearest neighbors using Z-Order space filling curve"""

    def __init__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        NNPS.__init__(
            self, dim, particles, radius_scale, ghost_layers, domain,
            cache, sort_gids
        )

        self.radius_scale2 = radius_scale*radius_scale
        cdef NNPSParticleArrayWrapper pa_wrapper
        cdef int i, num_particles

        for i from 0<=i<self.narrays:
            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.key_maps[i] = new key_to_idx_t()

        self.src_index = 0
        self.dst_index = 0
        self.sort_gids = sort_gids
        self.domain.update()
        self.update()

    def __cinit__(self, int dim, list particles, double radius_scale = 2.0,
            int ghost_layers = 1, domain=None, bint fixed_h = False,
            bint cache = False, bint sort_gids = False):
        cdef int narrays = len(particles)

        self.pids = <uint32_t**> malloc(narrays*sizeof(uint32_t*))
        self.keys = <uint64_t**> malloc(narrays*sizeof(uint64_t*))
        self.key_maps = <key_to_idx_t**> malloc(narrays*sizeof(key_to_idx_t*))

        self.current_pids = NULL
        self.current_map = NULL

    def __dealloc__(self):
        cdef int i
        for i from 0<=i<self.narrays:
            free(self.pids[i])
            free(self.keys[i])
            del self.key_maps[i]
        free(self.pids)
        free(self.keys)
        free(self.key_maps)

    cpdef set_context(self, int src_index, int dst_index):
        """Set context for nearest neighbor searches.

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        """
        NNPS.set_context(self, src_index, dst_index)
        self.current_pids = self.pids[src_index]
        self.current_map = self.key_maps[src_index]

        self.dst = <NNPSParticleArrayWrapper> self.pa_wrappers[dst_index]
        self.src = <NNPSParticleArrayWrapper> self.pa_wrappers[src_index]

    cdef void find_nearest_neighbors(self, size_t d_idx, UIntArray nbrs) nogil:
        """Low level, high-performance non-gil method to find neighbors.
        This requires that `set_context()` be called beforehand.  This method
        does not reset the neighbors array before it appends the
        neighbors to it.

        """
        cdef double* dst_x_ptr = self.dst.x.data
        cdef double* dst_y_ptr = self.dst.y.data
        cdef double* dst_z_ptr = self.dst.z.data
        cdef double* dst_h_ptr = self.dst.h.data

        cdef double* src_x_ptr = self.src.x.data
        cdef double* src_y_ptr = self.src.y.data
        cdef double* src_z_ptr = self.src.z.data
        cdef double* src_h_ptr = self.src.h.data

        cdef double x = dst_x_ptr[d_idx]
        cdef double y = dst_y_ptr[d_idx]
        cdef double z = dst_z_ptr[d_idx]
        cdef double h = dst_h_ptr[d_idx]

        cdef unsigned int* s_gid = self.src.gid.data
        cdef int orig_length = nbrs.length

        cdef int c_x, c_y, c_z
        cdef double* xmin = self.xmin.data
        cdef int i, j

        find_cell_id_raw(
                x - xmin[0],
                y - xmin[1],
                z - xmin[2],
                self.cell_size,
                &c_x, &c_y, &c_z
                )

        cdef double xij2 = 0
        cdef double hi2 = self.radius_scale2*h*h
        cdef double hj2 = 0

        cdef map[uint32_t, pair[uint32_t, uint32_t]].iterator it

        cdef int x_boxes[27]
        cdef int y_boxes[27]
        cdef int z_boxes[27]
        cdef int num_boxes = self._neighbor_boxes(c_x, c_y, c_z,
                x_boxes, y_boxes, z_boxes)

        cdef pair[uint32_t, uint32_t] candidate

        cdef uint32_t n, idx
        for i from 0<=i<num_boxes:
            it = self.current_map.find(get_key(x_boxes[i], y_boxes[i],
                z_boxes[i]))
            if it == self.current_map.end():
                continue
            candidate = deref(it).second
            n = candidate.first
            candidate_length = candidate.second

            for j from 0<=j<candidate_length:
                idx = self.current_pids[n+j]

                hj2 = self.radius_scale2*src_h_ptr[idx]*src_h_ptr[idx]

                xij2 = norm2(
                    src_x_ptr[idx] - x,
                    src_y_ptr[idx] - y,
                    src_z_ptr[idx] - z
                    )

                if (xij2 < hi2) or (xij2 < hj2):
                    nbrs.c_append(idx)

        if self.sort_gids:
            self._sort_neighbors(
                &nbrs.data[orig_length], nbrs.length - orig_length, s_gid
            )

    cpdef get_nearest_particles_no_cache(self, int src_index, int dst_index,
            size_t d_idx, UIntArray nbrs, bint prealloc):
        """Find nearest neighbors for particle id 'd_idx' without cache

        Parameters
        ----------
        src_index: int
            Index in the list of particle arrays to which the neighbors belong

        dst_index: int
            Index in the list of particle arrays to which the query point belongs

        d_idx: size_t
            Index of the query point in the destination particle array

        nbrs: UIntArray
            Array to be populated by nearest neighbors of 'd_idx'

        """
        self.set_context(src_index, dst_index)

        if prealloc:
            nbrs.length = 0
        else:
            nbrs.c_reset()

        self.find_nearest_neighbors(d_idx, nbrs)

    cpdef get_spatially_ordered_indices(self, int pa_index, LongArray indices):
        indices.reset()
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        cdef uint32_t* current_pids = self.pids[pa_index]

        cdef int j
        for j from 0<=j<num_particles:
            indices.c_append(<long>current_pids[j])

    cdef void fill_array(self, NNPSParticleArrayWrapper pa_wrapper, int pa_index,
            UIntArray indices, uint32_t* current_pids, uint64_t* current_keys,
            key_to_idx_t* current_map):
        cdef double* x_ptr = pa_wrapper.x.data
        cdef double* y_ptr = pa_wrapper.y.data
        cdef double* z_ptr = pa_wrapper.z.data

        cdef double* xmin = self.xmin.data

        cdef int c_x, c_y, c_z

        cdef int i, n
        for i from 0<=i<indices.length:
            find_cell_id_raw(
                    x_ptr[i] - xmin[0],
                    y_ptr[i] - xmin[1],
                    z_ptr[i] - xmin[2],
                    self.cell_size,
                    &c_x, &c_y, &c_z
                    )
            current_pids[i] = i
            current_keys[i] = get_key(c_x, c_y, c_z)

        cdef CompareSortWrapper sort_wrapper = \
                CompareSortWrapper(current_pids, current_keys, indices.length)

        sort_wrapper.compare_sort()

        cdef pair[uint32_t, pair[uint32_t, uint32_t]] temp
        cdef pair[uint32_t, uint32_t] cell, found_cell
        cdef map[uint64_t, pair[uint32_t, uint32_t]].iterator it

        temp.first = current_keys[0]
        cell.first = 0

        cdef uint32_t length = 0
        cdef uint32_t curr_cid = 0
        cdef uint32_t found_cid = UINT_MAX

        for i from 0<i<indices.length:
            length += 1

            if(current_keys[i] != current_keys[i-1]):

                for j from 0<=j<self.narrays:
                    iter_map = self.key_maps[j]
                    iter_cids = self.cids[j]
                    it = iter_map.find(temp.first)

                    if it != iter_map.end():
                        found_cell = deref(it).second
                        found_cid = iter_cids[found_cell.first]
                        break

                if found_cid == UINT_MAX:
                    curr_cid += 1

                current_cids[i] = curr_cid

                cell.second = length
                temp.second = cell
                current_map.insert(temp)

                temp.first = current_keys[i]
                cell.first = i

                length = 0

        cell.second = length + 1
        temp.second = cell
        current_map.insert(temp)

    cdef inline int _neighbor_boxes(self, int i, int j, int k,
            int* x, int* y, int* z) nogil:
        cdef int length = 0
        cdef int p, q, r
        for p from -1<=p<2:
            for q from -1<=q<2:
                for r from -1<=r<2:
                    if i+r>=0 and j+q>=0 and k+p>=0:
                        x[length] = i+r
                        y[length] = j+q
                        z[length] = k+p
                        length += 1
        return length

    cpdef _refresh(self):
        cdef NNPSParticleArrayWrapper pa_wrapper

        cdef int i, num_particles

        cdef double* xmax = self.xmax.data
        cdef double* xmin = self.xmin.data

        for i from 0<=i<self.narrays:
            free(self.pids[i])
            del self.key_maps[i]

            pa_wrapper = <NNPSParticleArrayWrapper> self.pa_wrappers[i]
            num_particles = pa_wrapper.get_number_of_particles()

            self.pids[i] = <uint32_t*> malloc(num_particles*sizeof(uint32_t))
            self.keys[i] = <uint64_t*> malloc(num_particles*sizeof(uint64_t))
            self.key_maps[i] = new key_to_idx_t()

        self.current_pids = self.pids[self.src_index]
        self.current_map = self.key_maps[self.src_index]

    @cython.cdivision(True)
    cpdef _bin(self, int pa_index, UIntArray indices):
        cdef NNPSParticleArrayWrapper pa_wrapper = self.pa_wrappers[pa_index]
        cdef int num_particles = pa_wrapper.get_number_of_particles()

        cdef uint32_t* current_pids = self.pids[pa_index]
        cdef uint64_t* current_keys = self.keys[pa_index]
        cdef key_to_idx_t* current_map = self.key_maps[pa_index]

        self.fill_array(pa_wrapper, pa_index, indices, current_pids, current_keys,
                current_map)

