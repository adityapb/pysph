import numpy as np
from mako.template import Template

import pysph.base.gpu_manager as gm
from pysph.base.particle_array import ParticleArray
from pysph.base.config import get_config


# args: uint* indices, dtype array, int length
REMOVE_KNL = Template(r"""
        unsigned int idx = indices[n - 1 - i];
        array[idx] = array[length - 1 - i];
""")


# args: tag_array, tag, indices, head
REMOVE_INDICES_KNL = Template(r"""
        if(tag_array[i] == tag)
            % if CUDA:
                indices[atomicAdd(&head[0], 1)] = i;
            % else:
                indices[atomic_inc(&head[0])] = i;
            % endif
""")


# args: tag_array, num_real_particles
NUM_REAL_PARTICLES_KNL = Template(r"""
        if(i != 0 && tag_array[i] != tag_array[i-1])
        {
            num_real_particles[0] = i;
            return;
        }
""")


class DeviceArray(object):
    def __init__(self, dtype, n=0):
        self.dtype = dtype
        length = n
        if n == 0:
            n = 16
        data = gm.empty(n, dtype)
        self.minimum = 0
        self.maximum = 0
        self.set_data(data)
        self.length = length
        self._update_array_ref()

    def _update_array_ref(self):
        self.array = self._data[:self.length]

    def resize(self, size):
        self.reserve(size)
        self.length = size
        self._update_array_ref()

    def reserve(self, size):
        if size > self.alloc:
            new_data = gm.empty(size, self.dtype)
            new_data[:self.alloc] = self._data
            self._data = new_data
            self.alloc = size
            self._update_array_ref()

    def set_data(self, data):
        self._data = data
        self.length = data.size
        self.alloc = data.size
        self.dtype = data.dtype
        self._update_array_ref()

    def get_data(self):
        return self._data

    def copy(self):
        arr_copy = DeviceArray(self.dtype)
        arr_copy.set_data(self.array.copy())
        return arr_copy

    def update_min_max(self):
        self.minimum = float(gm.gpu_array.min(self.array).get())
        self.maximum = float(gm.gpu_array.max(self.array).get())

    def fill(self, value):
        self.array.fill(value)

    def append(self, value):
        if self.length >= self.alloc:
            self.reserve(2 * self.length)
        self._data[self.length] = np.asarray(value, dtype=self.dtype)
        self.length += 1
        self._update_array_ref()

    # FIXME: change arg name to gpu_arr
    def extend(self, gpu_arr):
        if self.length + len(gpu_arr) > self.alloc:
            self.reserve(self.length + len(gpu_arr))
        self._data[-len(gpu_arr):] = gpu_arr
        self.length += len(gpu_arr)
        self._update_array_ref()

    def remove(self, indices, input_sorted=False):
        if len(indices) > self.length:
            return

        if not input_sorted:
            radix_sort = gm.get_radix_sort_kernel("unsigned int* indices",
                                                  "indices[i]", ["indices"])

            (sorted_indices,), event = radix_sort(indices)

        else:
            sorted_indices = indices

        args = "unsigned int* indices, %(dtype)s* array, unsigned int length" % \
                {"dtype" : gm.dtype_to_ctype(self.dtype)}
        src = REMOVE_KNL.render()
        remove = gm.get_elwise_kernel("remove", args, src)

        remove(sorted_indices, self.array, self.length)
        self.length -= len(indices)
        self._update_array_ref()

    def align(self, indices):
        self.set_data(gm.gpu_array.take(self.array, indices))

    def squeeze(self):
        self.set_data(self._data[:self.length])

    def copy_values(self, indices, dest):
        dest[:len(indices)] = gm.gpu_array.take(self.array, indices)


class DeviceHelper(object):
    """Manages the arrays contained in a particle array on the device.

    Note that it converts the data to a suitable type depending on the value of
    get_config().use_double. Further, note that it assumes that the names of
    constants and properties do not clash.

    """

    def __init__(self, particle_array):
        self._particle_array = pa = particle_array
        use_double = get_config().use_double
        self._dtype = np.float64 if use_double else np.float32
        self._data = {}
        self.properties = []
        self.constants = []

        for prop, ary in pa.properties.items():
            self.add_prop(prop, ary)
        for prop, ary in pa.constants.items():
            self.add_const(prop, ary)

    def _get_array(self, ary):
        ctype = ary.get_c_type()
        if ctype in ['float', 'double']:
            return ary.get_npy_array().astype(self._dtype)
        else:
            return ary.get_npy_array()

    def _get_prop_or_const(self, prop):
        pa = self._particle_array
        return pa.properties.get(prop, pa.constants.get(prop))

    def _add_prop_or_const(self, name, carray):
        """Add a new property or constant given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        np_array = self._get_array(carray)
        g_ary = DeviceArray(np_array.dtype, n=carray.length)
        g_ary.array.set(np_array)
        self._data[name] = g_ary
        setattr(self, name, g_ary.array)

    def _check_property(self, prop):
        """Check if a property is present or not """
        if prop in self.properties:
            return
        else:
            raise AttributeError('property %s not present' % (prop))

    def get_number_of_particles(self, real=False):
        if real:
            return self.num_real_particles
        else:
            if len(self.properties) > 0:
                prop0 = self._data[self.properties[0]]
                return len(prop0.array)
            else:
                return 0

    def align(self, indices):
        for prop in self.properties:
            self._data[prop].align(indices)
            setattr(self, prop, self._data[prop].array)

    def add_prop(self, name, carray):
        """Add a new property given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        self._add_prop_or_const(name, carray)
        if name in self._particle_array.properties:
            self.properties.append(name)

    def add_const(self, name, carray):
        """Add a new constant given the name and carray, note
        that this assumes that this property is already added to the
        particle array.
        """
        self._add_prop_or_const(name, carray)
        if name in self._particle_array.constants:
            self.constants.append(name)

    def update_prop(self, name, dev_array):
        """Add a new property to DeviceHelper. Note that this property
        is not added to the particle array itself"""
        self._data[name] = dev_array
        setattr(self, name, dev_array.array)
        if name not in self.properties:
            self.properties.append(name)

    def update_const(self, name, dev_array):
        """Add a new constant to DeviceHelper. Note that this property
        is not added to the particle array itself"""
        self._data[name] = dev_array
        setattr(self, name, dev_array.array)
        if name not in self.constants:
            self.constants.append(name)

    def get_device_array(self, prop):
        if prop in self.properties or prop in self.constants:
            return self._data[prop]

    def max(self, arg):
        return float(gm.gpu_array.max(getattr(self, arg)).get())

    def update_min_max(self, props=None):
        """Update the min,max values of all properties """
        if props:
            for prop in props:
                array = self._data[prop]
                array.update_min_max()
        else:
            for prop in self.properties:
                array = self._data[prop]
                array.update_min_max()

    def pull(self, *args):
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            self._get_prop_or_const(arg).set_data(
                getattr(self, arg).get()
            )

    def push(self, *args):
        if len(args) == 0:
            args = self._data.keys()
        for arg in args:
            getattr(self, arg).set(
                self._get_array(self._get_prop_or_const(arg))
            )

    def remove_prop(self, name):
        if name in self.properties:
            self.properties.remove(name)
        if name in self._data:
            del self._data[name]
            delattr(self, name)

    def resize(self, new_size):
        for prop in self.properties:
            self._data[prop].resize(new_size)
            setattr(self, prop, self._data[prop].array)

    def align_particles(self):
        tag_arr = self._data['tag'].array

        num_particles = self.get_number_of_particles()
        indices = gm.arange(0, num_particles, 1,
                dtype=np.uint32)

        radix_sort = gm.get_radix_sort_kernel("unsigned int* indices, unsigned int* tags",
                                            "tags[i]", ["indices"])

        (sorted_indices,), event = radix_sort(indices, tag_arr)
        self.align(sorted_indices)

        tag_arr = self._data['tag'].array

        num_real_particles = gm.zeros(1, np.uint32)
        args = "unsigned int* tag_array, unsigned int* num_real_particles"
        src = NUM_REAL_PARTICLES_KNL.render()
        get_num_real_particles = gm.get_elwise_kernel("get_num_real_particles", args, src)

        get_num_real_particles(tag_arr, num_real_particles)
        self.num_real_particles = int(num_real_particles.get())

    def remove_particles(self, indices):
        """ Remove particles whose indices are given in index_list.

        We repeatedly interchange the values of the last element and values from
        the index_list and reduce the size of the array by one. This is done for
        every property that is being maintained.

        Parameters
        ----------

        indices : array
            an array of indices, this array can be a list, numpy array
            or a LongArray.

        Notes
        -----

        Pseudo-code for the implementation::

            if index_list.length > number of particles
                raise ValueError

            sorted_indices <- index_list sorted in ascending order.

            for every every array in property_array
                array.remove(sorted_indices)

        """
        if len(indices) > self.get_number_of_particles():
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError(msg)

        radix_sort = gm.get_radix_sort_kernel("unsigned int* indices",
                                           "indices[i]", ["indices"])

        (sorted_indices,), event = radix_sort(indices)

        for prop in self.properties:
            self._data[prop].remove(sorted_indices, 1)
            setattr(self, prop, self._data[prop].array)

        if len(indices) > 0:
            self.align_particles()

    def remove_tagged_particles(self, tag):
        """ Remove particles that have the given tag.

        Parameters
        ----------

        tag : int
            the type of particles that need to be removed.

        """
        tag_array = getattr(self, 'tag')

        remove_places = tag_array == tag
        num_indices = int(gm.gpu_array.sum(remove_places).get())

        if num_indices == 0:
            return

        indices = gm.empty(num_indices, np.uint32)
        head = gm.zeros(1, np.uint32)

        args = "unsigned int* tag_array, unsigned int tag, \
                unsigned int* indices, unsigned int* head"
        src = REMOVE_INDICES_KNL.render(CUDA=get_config().use_cuda)

        # find the indices of the particles to be removed.
        remove_indices = gm.get_elwise_kernel("remove_indices", args, src)

        remove_indices(tag_array, tag, indices, head)

        # remove the particles.
        self.remove_particles(indices)

    def add_particles(self, **particle_props):
        """
        Add particles in particle_array to self.

        Parameters
        ----------

        particle_props : dict
            a dictionary containing cl arrays for various particle
            properties.

        Notes
        -----

         - all properties should have same length arrays.
         - all properties should already be present in this particles array.
           if new properties are seen, an exception will be raised.
           properties.

        """
        pa = self._particle_array

        if len(particle_props) == 0:
            return

        # check if the input properties are valid.
        for prop in particle_props:
            self._check_property(prop)

        num_extra_particles = len(list(particle_props.values())[0])
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        for prop in self.properties:
            arr = self._data[prop]

            if prop in particle_props.keys():
                s_arr = particle_props[prop]
                arr.extend(s_arr)
            else:
                arr.resize(new_num_particles)
                # set the properties of the new particles to the default ones.
                arr.array[old_num_particles:] = np.ones(num_extra_particles,
                        dtype=arr.dtype) + pa.default_values[prop]

            self.update_prop(prop, arr)

        if num_extra_particles > 0:
            # make sure particles are aligned properly.
            self.align_particles()

    def extend(self, num_particles):
        """ Increase the total number of particles by the requested amount

        New particles are added at the end of the list, you may have to manually
        call align_particles later.
        """
        if num_particles <= 0:
            return

        old_size = self.get_number_of_particles()
        new_size = old_size + num_particles

        for prop in self.properties:
            arr = self._data[prop]
            arr.resize(new_size)
            arr.array[old_size:] = np.ones(num_particles, dtype=arr.dtype) + \
                    self._particle_array.default_values[prop]
            self.update_prop(prop, arr)

    def append_parray(self, parray):
        """ Add particles from a particle array

        properties that are not there in self will be added
        """
        if parray.gpu is None:
            parray.set_device_helper(DeviceHelper(parray))

        if parray.gpu.get_number_of_particles() == 0:
            return

        num_extra_particles = parray.gpu.get_number_of_particles()
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        # extend current arrays by the required number of particles
        self.extend(num_extra_particles)

        for prop_name in parray.gpu.properties:
            if prop_name in self.properties:
                arr = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                arr.array[old_num_particles:] = source.array
            else:
                # meaning this property is not there in self.
                dtype = self._data[prop_name].dtype
                arr = DeviceArray(dtype, n=new_num_particles)
                arr.fill(parray.gpu._particle_array.default_values[prop_name])
                self.update_prop(prop_name, arr)

                # now add the values to the end of the created array
                dest = self._data[prop_name]
                source = parray.gpu.get_device_array(prop_name)
                dest.array[old_num_particles:] = source.array

        for const in parray.gpu.constants:
            if const not in self.constants:
                arr = parray.gpu.get_device_array(const)
                self.update_const(const, arr.copy())

        if num_extra_particles > 0:
            self.align_particles()

    def extract_particles(self, indices, props=None):
        """Create new particle array for particles with given indices

        Parameters
        ----------

        indices : cl.array.Array
            indices of particles to be extracted.

        props : list
            the list of properties to extract, if None all properties
            are extracted.

        """
        result_array = ParticleArray()
        result_array.set_device_helper(DeviceHelper(result_array))

        if props is None:
            prop_names = self.properties
        else:
            prop_names = props

        if len(indices) == 0:
            return result_array

        for prop_name in prop_names:
            src_arr = self._data[prop_name]
            dst_arr = DeviceArray(src_arr.dtype, n=len(indices))
            src_arr.copy_values(indices, dst_arr.array)

            prop_type = gm.dtype_to_ctype(src_arr.dtype)
            prop_default = self._particle_array.default_values[prop_name]
            result_array.add_property(name=prop_name,
                                      type=prop_type,
                                      default=prop_default)

            result_array.gpu.update_prop(prop_name, dst_arr)

        for const in self.constants:
            result_array.gpu.update_const(const, self._data[const].copy())

        result_array.gpu.align_particles()
        result_array.set_name(self._particle_array.name)

        if props is None:
            output_arrays = list(self._particle_array.output_property_arrays)
        else:
            output_arrays = list(
                set(props).intersection(
                    self._particle_array.output_property_arrays
                )
            )

        result_array.set_output_arrays(output_arrays)
        return result_array
