import numpy as np
import pyopencl as cl
import pyopencl.array

from pysph.base.nnps_base import DomainManagerBase
from pysph.base.opencl import get_config

from pysph.cpy.array import Array, get_backend
from pysph.cpy.parallel import Elementwise
from pysph.cpy.types import annotate, dtype_to_ctype
from pysph.cpy


class GPUDomainManager(DomainManagerBase):
    def __init__(self, xmin=-1000., xmax=1000., ymin=0.,
                 ymax=0., zmin=0., zmax=0.,
                 periodic_in_x=False, periodic_in_y=False,
                 periodic_in_z=False, n_layers=2.0):
        """Constructor"""
        DomainManagerBase.__init__(xmin=xmin, xmax=xmax,
                ymin=ymin, ymax=ymax, zmin=zmin, zmax=zmax,
                periodic_in_x=periodic_in_x, periodic_in_y=periodic_in_y,
                periodic_in_z=periodic_in_z, n_layers=n_layers)

        self.use_double = get_config().use_double
        self.dtype = np.float64 if use_double else np.float32

        self.dtype_max = np.finfo(self.dtype).max
        self.backend = get_backend()

    def update(self, *args, **kwargs):
        """General method that is called before NNPS can bin particles.

        This method is responsible for the computation of cell sizes
        and creation of any ghost particles for periodic or wall
        boundary conditions.

        """
        # compute the cell sizes
        self.compute_cell_size_for_binning()

        # Periodicity is handled by adjusting particles according to a
        # given cubic domain box. In parallel, it is expected that the
        # appropriate parallel NNPS is responsible for the creation of
        # ghost particles.
        if self.is_periodic and not self.in_parallel:
            self._update_from_gpu()

            # remove periodic ghost particles from a previous step
            self._remove_ghosts()

            # box-wrap current particles for periodicity
            self._box_wrap_periodic()

            # create new periodic ghosts
            self._create_ghosts_periodic()

            # Update GPU.
            self._update_gpu()

    def _compute_cell_size_for_binning(self):
        """Compute the cell size for the binning.

        The cell size is chosen as the kernel radius scale times the
        maximum smoothing length in the local processor. For parallel
        runs, we would need to communicate the maximum 'h' on all
        processors to decide on the appropriate binning size.

        """
        _hmax, hmax = -1.0, -1.0
        _hmin, hmin = self.dtype_max, self.dtype_max

        for pa_wrapper in self.pa_wrappers:
            h = pa_wrapper.pa.gpu.get_device_array('h')
            h.update_min_max()

            _hmax = h.maximum
            _hmin = h.minimum

            if _hmax > hmax:
                hmax = _hmax
            if _hmin < hmin:
                hmin = _hmin

        cell_size = self.radius_scale * hmax
        self.hmin = self.radius_scale * hmin

        if cell_size < 1e-6:
            cell_size = 1.0

        self.cell_size = cell_size

        # set the cell size for the DomainManager
        self.set_cell_size(cell_size)

    def _get_box_wrap_kernel(self):
        val_type = dtype_to_ctype(self.dtype)
        ptr_type = 'g%sp' % val_type

        types = {ptr_type : 'x, y, z',
                 val_type : 'xtranslate, ytranslate, ztranslate',
                 'int' : 'periodic_int_x, periodic_in_y, periodic_in_z'}

        @annotate(**types)
        def box_wrap(x, y, z, xtranslate, ytranslate, ztranslate,
                     periodic_in_x, periodic_in_y, periodic_in_z):
            if periodic_in_x:
                if x[i] < xmin : x[i] = x[i] + xtranslate
                if x[i] > xmax : x[i] = x[i] - xtranslate

            if periodic_in_y:
                if y[i] < ymin : y[i] = y[i] + ytranslate
                if y[i] > ymax : y[i] = y[i] - ytranslate

            if periodic_in_z:
                if z[i] < zmin : z[i] = z[i] + ztranslate
                if z[i] > zmax : z[i] = z[i] - ztranslate

        return Elementwise(box_wrap, backend=self.backend)


    ###########################CHANGE FROM HERE####################################

    def _box_wrap_periodic(self):
        """Box-wrap particles for periodicity

        The periodic domain is a rectangular box defined by minimum
        and maximum values in each coordinate direction. These values
        are used in turn to define translation values used to box-wrap
        particles that cross a periodic boundary.

        The periodic domain is specified using the DomainManager object

        """
        # minimum and maximum values of the domain
        xmin = self.xmin, xmax = self.xmax
        ymin = self.ymin, ymax = self.ymax,
        zmin = self.zmin, zmax = self.zmax

        # translations along each coordinate direction
        xtranslate = self.xtranslate
        ytranslate = self.ytranslate
        ztranslate = self.ztranslate

        # periodicity flags for NNPS
        periodic_in_x = self.periodic_in_x
        periodic_in_y = self.periodic_in_y
        periodic_in_z = self.periodic_in_z

        # iterate over each array and mark for translation
        for pa_wrapper in self.pa_wrappers:
            x = Array(); y = Array(); z = Array()

            x.set_dev_array(pa_wrapper.gpu.x)
            y.set_dev_array(pa_wrapper.gpu.y)
            z.set_dev_array(pa_wrapper.gpu.z)

            box_wrap_knl = self._get_box_wrap_kernel()

            box_wrap_knl(x, y, z, xtranslate, ytranslate, ztranslate,
                         periodic_in_x, periodic_in_y, periodic_in_z)

    def _check_limits(self, xmin, xmax, ymin, ymax, zmin, zmax):
        """Sanity check on the limits"""
        if ( (xmax < xmin) or (ymax < ymin) or (zmax < zmin) ):
            raise ValueError("Invalid domain limits!")

    def _create_ghosts_periodic(self):
        """Identify boundary particles and create images.

        We need to find all particles that are within a specified
        distance from the boundaries and place image copies on the
        other side of the boundary. Corner reflections need to be
        accounted for when using domains with multiple periodicity.

        The periodic domain is specified using the DomainManager object

        """
        pa_wrappers = self.pa_wrappers
        narrays = self.narrays

        # cell size used to check for periodic ghosts. For summation density
        # like operations, we need to create two layers of ghost images, this
        # is configurable via the n_layers argument to the constructor.
        cell_size = self.n_layers * self.cell_size

        # periodic domain values
        xmin = self.xmin, xmax = self.xmax
        ymin = self.ymin, ymax = self.ymax,
        zmin = self.zmin, zmax = self.zmax

        xtranslate = self.xtranslate
        ytranslate = self.ytranslate
        ztranslate = self.ztranslate

        # periodicity flags
        periodic_in_x = self.periodic_in_x
        periodic_in_y = self.periodic_in_y
        periodic_in_z = self.periodic_in_z

        # locals
        #cdef NNPSParticleArrayWrapper pa_wrapper
        #cdef ParticleArray pa, added
        #cdef DoubleArray x, y, z, xt, yt, zt
        #cdef double xi, yi, zi
        #cdef int array_index, i, np

        # temporary indices for particles to be replicated
        #cdef LongArray x_low, x_high, y_high, y_low, z_high, z_low, low, high

        x_low = LongArray(); x_high = LongArray()
        y_high = LongArray(); y_low = LongArray()
        z_high = LongArray(); z_low = LongArray()
        low = LongArray(); high = LongArray()

        for array_index in range(narrays):
            pa_wrapper = pa_wrappers[ array_index ]
            pa = pa_wrapper.pa
            x = pa_wrapper.x; y = pa_wrapper.y; z = pa_wrapper.z

            # reset the length of the arrays
            x_low.reset(); x_high.reset(); y_high.reset(); y_low.reset()
            z_low.reset(); z_high.reset()

            np = x.length
            for i in range(np):
                xi = x.data[i]; yi = y.data[i]; zi = z.data[i]

                if periodic_in_x:
                    if ( (xi - xmin) <= cell_size ): x_low.append(i)
                    if ( (xmax - xi) <= cell_size ): x_high.append(i)

                if periodic_in_y:
                    if ( (yi - ymin) <= cell_size ): y_low.append(i)
                    if ( (ymax - yi) <= cell_size ): y_high.append(i)

                if periodic_in_z:
                    if ( (zi - zmin) <= cell_size ): z_low.append(i)
                    if ( (zmax - zi) <= cell_size ): z_high.append(i)


            # now treat each case separately and append to the main array
            added = ParticleArray(x=None, y=None, z=None)
            x = added.get_carray('x')
            y = added.get_carray('y')
            z = added.get_carray('z')
            if periodic_in_x:
                # x_low
                copy = pa.extract_particles( x_low )
                self._add_to_array(copy.get_carray('x'), xtranslate)
                added.append_parray(copy)

                # x_high
                copy = pa.extract_particles( x_high )
                self._add_to_array(copy.get_carray('x'), -xtranslate)
                added.append_parray(copy)

            if periodic_in_y:
                # Now do the corners from the previous.
                low.reset(); high.reset()
                np = x.length
                for i in range(np):
                    yi = y.data[i]
                    if ( (yi - ymin) <= cell_size ): low.append(i)
                    if ( (ymax - yi) <= cell_size ): high.append(i)

                copy = added.extract_particles(low)
                self._add_to_array(copy.get_carray('y'), ytranslate)
                added.append_parray(copy)

                copy = added.extract_particles(high)
                self._add_to_array(copy.get_carray('y'), -ytranslate)
                added.append_parray(copy)

                # Add the actual y_high and y_low now.
                # y_high
                copy = pa.extract_particles( y_high )
                self._add_to_array(copy.get_carray('y'), -ytranslate)
                added.append_parray(copy)

                # y_low
                copy = pa.extract_particles( y_low )
                self._add_to_array(copy.get_carray('y'), ytranslate)
                added.append_parray(copy)

            if periodic_in_z:
                # Now do the corners from the previous.
                low.reset(); high.reset()
                np = x.length
                for i in range(np):
                    zi = z.data[i]
                    if ( (zi - zmin) <= cell_size ): low.append(i)
                    if ( (zmax - zi) <= cell_size ): high.append(i)

                copy = added.extract_particles(low)
                self._add_to_array(copy.get_carray('z'), ztranslate)
                added.append_parray(copy)

                copy = added.extract_particles(high)
                self._add_to_array(copy.get_carray('z'), -ztranslate)
                added.append_parray(copy)

                # Add the actual z_high and z_low now.
                # z_high
                copy = pa.extract_particles( z_high )
                self._add_to_array(copy.get_carray('z'), -ztranslate)
                added.append_parray(copy)

                # z_low
                copy = pa.extract_particles( z_low )
                self._add_to_array(copy.get_carray('z'), ztranslate)
                added.append_parray(copy)


            added.tag[:] = Ghost
            pa.append_parray(added)


