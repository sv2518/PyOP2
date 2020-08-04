# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""OP2 OpenCL backend."""

import os
from copy import deepcopy as dcopy

from hashlib import md5
from contextlib import contextmanager

from pyop2.datatypes import IntType, as_ctypes
from pyop2 import base
from pyop2 import compilation
from pyop2 import petsc_base
from pyop2 import sequential
from pyop2.exceptions import *  # noqa: F401
from pyop2.mpi import collective
from pyop2.profiling import timed_region
from pyop2.utils import *
from pyop2.configuration import configuration

import numpy
from pytools import memoize_method
from pyop2.petsc_base import PETSc, AbstractPETScBackend
from pyop2.utils import cached_property

import pyopencl as cl


def cl_buffer_from_numpy_array(ary, mode='rw'):
    mf = cl.mem_flags
    if mode == 'r':
        access_flg = mf.READ_ONLY
    else:
        assert mode == 'rw'
        access_flg = mf.READ_WRITE

    ctx = opencl_backend.context
    ary_on_device = cl.Buffer(ctx, access_flg | mf.COPY_HOST_PTR, hostbuf=ary)
    return ary_on_device


class Map(base.Map):

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        self._opencl_values = None

    def get_availability(self):
        if self._opencl_values is None:
            return base.AVAILABLE_ON_HOST_ONLY

        return base.AVAILABLE_ON_BOTH

    def ensure_availability_on_device(self):
        if self._opencl_values is None:
            self._opencl_values = cl_buffer_from_numpy_array(self._values, 'r')

    def ensure_availability_on_host(self):
        # Map once initialized is not over-written so always available
        # on host.
        pass

    def is_available_on_device(self):
        return bool(self.get_availability() & base.AVAILABLE_ON_DEVICE_ONLY)

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Map unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            return (self._opencl_values,)
        else:
            return super(Map, self)._kernel_args_


class ExtrudedSet(base.ExtrudedSet):
    """
    ExtrudedSet for OpenCL.
    """

    def __init__(self, *args, **kwargs):
        super(ExtrudedSet, self).__init__(*args, **kwargs)
        self.opencl_layers_array = None

    def get_availability(self):
        if self.opencl_layers_array is None:
            return base.AVAILABLE_ON_HOST_ONLY

        return base.AVAILABLE_ON_BOTH

    def ensure_availability_on_device(self):
        if self.opencl_layers_array is None:
            self.opencl_layers_array = cl_buffer_from_numpy_array(self.layers_array, 'r')

    def ensure_availability_on_host(self):
        # ExtrudedSet once initialized is not over-written so always available
        # on host.
        pass

    def is_available_on_device(self):
        return bool(self.get_availability() & base.AVAILABLE_ON_DEVICE_ONLY)

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("ExtrudedSet unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            return (self.opencl_layers_array,)
        else:
            return super(ExtrudedSet, self)._kernel_args_


class Subset(base.Subset):
    """
    Subset for OpenCL.
    """
    def __init__(self, *args, **kwargs):
        super(Subset, self).__init__(*args, **kwargs)
        self._opencl_indices = None

    def get_availability(self):
        if self._opencl_indices is None:
            return base.AVAILABLE_ON_HOST_ONLY

        return base.AVAILABLE_ON_BOTH

    def ensure_availability_on_device(self):
        if self._opencl_indices is None:
            self._opencl_indices = cl_buffer_from_numpy_array(self._indices, 'r')

    def ensure_availability_on_host(self):
        # Subset once initialized is not over-written so always available
        # on host.
        pass

    def is_available_on_device(self):
        return bool(self.get_availability() & base.AVAILABLE_ON_DEVICE_ONLY)

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Subset unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            return (self._opencl_indices,)
        else:
            return super(Subset, self)._kernel_args_


class DataSet(petsc_base.DataSet):
    """
    At the moment I don't think there should be any over-riding needed.
    """


class Dat(petsc_base.Dat):
    """
    Dat for OpenCL.
    """
    def __init__(self, *args, **kwargs):
        super(Dat, self).__init__(*args, **kwargs)
        # opencl_data, offload_mask: used only when cannot be represented as petscvec
        self._opencl_data = cl_buffer_from_numpy_array(self._data, 'rw')
        self.availability_flag = base.AVAILABLE_ON_BOTH

    @contextmanager
    def vec_context(self, access):
        r"""A context manager for a :class:`PETSc.Vec` from a :class:`Dat`.

        :param access: Access descriptor: READ, WRITE, or RW."""
        # PETSc Vecs have a state counter and cache norm computations
        # to return immediately if the state counter is unchanged.
        # Since we've updated the data behind their back, we need to
        # change that state counter.
        self._vec.stateIncrease()

        if opencl_backend.offloading:
            self._vec.bindToCPU(False)
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Dat unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()
        else:
            if not self.is_available_on_host():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Dat unavailable on host. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_host()
            self._vec.bindToCPU(True)

        yield self._vec

        if access is not base.READ:
            self.halo_valid = False
            self._vec.restoreCLMemHandle()  # let petsc know that GPU data has been altered

    def get_availability(self):
        # FIXME: THis needs to be fixed
        return base.AVAILABLE_ON_HOST_ONLY
        if self.can_be_represented_as_petscvec():
            with self.vec_ro as v:
                return DataAvailability(v.getOffloadMask())
        else:
            return self.availability_flag

    def ensure_availability_on_device(self):
        if self.can_be_represented_as_petscvec():
            self._vec.getCLMemHandle('r')  # performs a host->device transfer if needed
        else:
            if not self.is_available_on_host():
                cl.enqueue_copy(opencl_backend.queue, self._opencl_data, self._data)
            self.availability_flag = AVAILABLE_ON_BOTH

    def ensure_availability_on_host(self):
        if self.can_be_represented_as_petscvec():
            self._vec.getArray(readonly=True)  # performs a device->host transfer if needed
        else:
            if not self.is_available_on_host():
                cl.enqueue_copy(opencl_backend.queue, self._data, self._opencl)
            self.availability_flag = AVAILABLE_ON_BOTH

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            import pudb; pu.db
        if self.can_be_represented_as_petscvec():
            with self.vec as v:
                if opencl_backend.offloading:
                    return (cl.MemoryObject.from_int_ptr(v.getCLMemHandle('rw'), False), )
                else:
                    return (self._data.ctypes.data, )
        else:
            if opencl_backend.offloading:
                if not self.is_available_on_device():
                    if configuration['only_explicit_host_device_data_transfers']:
                        raise RuntimeError("Dat unavailable on device. Call"
                                           " ensure_availability_on_device()")

                    self.ensure_availability_on_device()

                self.availability_flag = AVAILABLE_ON_DEVICE_ONLY
                return (self._opencl_data, )
            else:
                if not self.is_available_on_host():
                    if configuration['only_explicit_host_device_data_transfers']:
                        raise RuntimeError("Dat unavailable on host. Call"
                                           " ensure_availability_on_device()")

                    self.ensure_availability_on_host()

                self.availability_flag = AVAILABLE_ON_HOST_ONLY
                return (self._data.ctypes.data, )

    @collective
    @property
    def data(self):
        if self.dataset.total_size > 0 and self._data.size == 0 and self.cdim > 0:
            raise RuntimeError("Illegal access: no data associated with this Dat!")
        self.halo_valid = False

        if not self.is_available_on_host():
            if configuration['only_explicit_host_device_data_transfers']:
                raise RuntimeError("Dat unavailable on host. Call"
                                   " ensure_availability_on_device()")
            self.ensure_availability_on_host()

        v = self._data[:self.dataset.size].view()
        v.setflags(write=True)

        # {{{ marking data on the device as invalid

        if self.can_be_represented_as_petsvec():
            pass
            # self._vec.getArray()
        else:
            self.availability_flag = AVAILABLE_ON_HOST_ONLY

        # }}}

        return v


class Global(petsc_base.Global):
    """
    Global for OpenCL.
    """

    def __init__(self, *args, **kwargs):
        super(Global, self).__init__(*args, **kwargs)
        self._opencl_data = None

    def get_availability(self):
        if self._opencl_data is None:
            return base.AVAILABLE_ON_HOST_ONLY

        return base.AVAILABLE_ON_BOTH

    def ensure_availability_on_device(self):
        if self._opencl_data is None:
            self._opencl_data = cl_buffer_from_numpy_array(self._data, 'r')

    def ensure_availability_on_host(self):
        # Global once initialized is not over-written so always available
        # on host.
        pass

    def is_available_on_device(self):
        return bool(self.get_availability() & base.AVAILABLE_ON_DEVICE_ONLY)

    @property
    def _kernel_args_(self):
        if opencl_backend.offloading:
            if not self.is_available_on_device():
                if configuration['only_explicit_host_device_data_transfers']:
                    raise RuntimeError("Global unavailable on device. Call"
                                       " ensure_availability_on_device()")

                self.ensure_availability_on_device()

            return (self._opencl_data,)
        else:
            return super(Global, self)._kernel_args_


class JITModule(base.JITModule):

    _cppargs = []
    _libraries = []
    _system_headers = []

    def __init__(self, kernel, iterset, *args, **kwargs):
        r"""
        A cached compiled function to execute for a specified par_loop.

        See :func:`~.par_loop` for the description of arguments.

        .. warning ::

           Note to implementers.  This object is *cached*, and therefore
           should not hold any long term references to objects that
           you want to be collected.  In particular, after the
           ``args`` have been inspected to produce the compiled code,
           they **must not** remain part of the object's slots,
           otherwise they (and the :class:`~.Dat`\s, :class:`~.Map`\s
           and :class:`~.Mat`\s they reference) will never be collected.
        """
        # Return early if we were in the cache.
        if self._initialized:
            return
        self.comm = iterset.comm
        self._kernel = kernel
        self._fun = None
        self._iterset = iterset
        self._args = args
        self._iteration_region = kwargs.get('iterate', base.ALL)
        self._pass_layer_arg = kwargs.get('pass_layer_arg', False)
        # Copy the class variables, so we don't overwrite them
        self._cppargs = dcopy(type(self)._cppargs)
        self._libraries = dcopy(type(self)._libraries)
        self._system_headers = dcopy(type(self)._system_headers)
        if not kwargs.get('delay', False):
            self.compile()
            self._initialized = True

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        key = super(JITModule, cls)._cache_key(*args, **kwargs)
        key += (configuration["gpu_strategy"],)
        if configuration["gpu_strategy"] == "scpt":
            pass
        elif configuration["gpu_strategy"] == "user_specified_tile":
            key += (configuration["gpu_cells_per_block"],
                    configuration["gpu_threads_per_cell"],
                    configuration["gpu_op_tile_descriptions"],
                    configuration["gpu_quad_rowtile_lengths"],
                    configuration["gpu_input_to_shared"],
                    configuration["gpu_quad_weights_to_shared"],
                    configuration["gpu_mats_to_shared"],
                    configuration["gpu_tiled_prefetch_of_input"],
                    configuration["gpu_tiled_prefetch_of_quad_weights"],)
        elif configuration["gpu_strategy"] == "auto_tile":
            key += (configuration["gpu_planner_kernel_evals"],)
        else:
            raise NotImplementedError('For strategy: {}'.format(
                configuration["gpu_strategy"]))
        return key

    @memoize_method
    def grid_size(self, start, end):
        with open(self.config_file_path, 'r') as f:
            glens_llens = f.read()

        _, glens, llens = glens_llens.split('\n')
        from pymbolic import parse, evaluate
        glens = parse(glens)
        llens = parse(llens)

        parameters = {'start': start, 'end': end}

        grid = tuple(int(evaluate(glens[i], parameters)) if i < len(glens) else 1
                     for i in range(2))
        block = tuple(int(evaluate(llens[i], parameters)) if i < len(llens) else 1
                      for i in range(3))

        return grid, block

    @cached_property
    def get_args_marked_for_globals(self):
        args_to_make_global = []
        for i in range(self.num_args_to_make_global):
            args_to_make_global.append(numpy.load(self.ith_added_global_arg_i(i)))

        const_args_as_globals = tuple(cl_buffer_from_numpy_array(arg, 'r')
                                      for arg in args_to_make_global)

        return const_args_as_globals

    @cached_property
    def config_file_path(self):
        cachedir = configuration['cache_dir']
        return os.path.join(cachedir, '{}_num_args_to_load_glens_llens_opencl'.format(self.get_encoded_cache_key))

    @memoize_method
    def ith_added_global_arg_i(self, i):
        cachedir = configuration['cache_dir']
        return os.path.join(cachedir, '{}_opencl_dat_{}.npy'.format(self.get_encoded_cache_key, i))

    @collective
    def __call__(self, *args):
        if self._initialized:
            grid, block = self.grid_size(args[0], args[1])
            extra_global_args = self.get_args_marked_for_globals
        else:
            self.args = args[:]
            self.compile()
            self._initialized = True
            return self.__call__(*args)

        Au_before = np.empty(121)
        coords_before = np.empty((121, 2))
        u_before = np.empty(121)
        cl.enqueue_copy(opencl_backend.queue, Au_before, args[2])
        cl.enqueue_copy(opencl_backend.queue, coords_before, args[3])
        cl.enqueue_copy(opencl_backend.queue, u_before, args[4])
        import pudb; pu.db

        self._fun(opencl_backend.queue, grid, block, *(args+extra_global_args), g_times_l=True)

    @cached_property
    def _wrapper_name(self):
        return 'wrap_%s' % self._kernel.name

    @cached_property
    def num_args_to_make_global(self):
        with open(self.config_file_path, 'r') as f:
            return int(f.readline().strip())

    @cached_property
    def get_encoded_cache_key(self):
        a = md5(str(self.cache_key).encode()).hexdigest()
        return a

    @cached_property
    def code_to_compile(self):
        assert self.args is not None
        from pyop2.codegen.builder import WrapperBuilder
        from pyop2.codegen.rep2loopy import generate
        from pyop2.gpu.generate import generate_gpu_kernel

        builder = WrapperBuilder(iterset=self._iterset, iteration_region=self._iteration_region, pass_layer_to_kernel=self._pass_layer_arg)
        for arg in self._args:
            builder.add_argument(arg)
        builder.set_kernel(self._kernel)

        wrapper = generate(builder, include_math=False, include_petsc=False, include_complex=False)

        code, processed_program, args_to_make_global = generate_gpu_kernel(wrapper, self.args, self.argshapes, 'opencl')
        for i, arg_to_make_global in enumerate(args_to_make_global):
            numpy.save(self.ith_added_global_arg_i(i),
                       arg_to_make_global)

        with open(self.config_file_path, 'w') as f:
            glens, llens = processed_program.get_grid_size_upper_bounds_as_exprs()
            f.write(str(len(args_to_make_global)))
            f.write('\n')
            f.write('('+','.join(str(glen) for glen in glens)+',)')
            f.write('\n')
            f.write('('+','.join(str(llen) for llen in llens)+',)')

        return code

    @collective
    def compile(self):

        # If we weren't in the cache we /must/ have arguments
        if not hasattr(self, '_args'):
            raise RuntimeError("JITModule has no args associated with it, should never happen")

        compiler = "opencl"
        extension = "cl"
        self._fun = compilation.load(self,
                                     extension,
                                     self._wrapper_name,
                                     cppargs=[],
                                     ldargs=[],
                                     compiler=compiler,
                                     comm=self.comm)

        self._fun.set_scalar_arg_dtypes([IntType, IntType]
                                        + [None]*(len(self.argtypes[2:])+self.num_args_to_make_global))

        # Blow away everything we don't need any more
        del self.args
        del self._args
        del self._kernel
        del self._iterset

    @cached_property
    def argtypes(self):
        index_type = as_ctypes(IntType)
        argtypes = (index_type, index_type)
        argtypes += self._iterset._argtypes_
        for arg in self._args:
            argtypes += arg._argtypes_
        seen = set()
        for arg in self._args:
            maps = arg.map_tuple
            for map_ in maps:
                for k, t in zip(map_._kernel_args_, map_._argtypes_):
                    if k in seen:
                        continue
                    argtypes += (t,)
                    seen.add(k)
        return argtypes

    @cached_property
    def argshapes(self):
        argshapes = ((), ())
        if self._iterset._argtypes_:
            # FIXME: Do not put in a bogus value
            argshapes += ((), )

        for arg in self._args:
            argshapes += (arg.data.shape, )
        seen = set()
        for arg in self._args:
            maps = arg.map_tuple
            for map_ in maps:
                for k, t in zip(map_._kernel_args_, map_._argtypes_):
                    if k in seen:
                        continue
                    argshapes += (map_.shape, )
                    seen.add(k)

        return argshapes


class ParLoop(petsc_base.ParLoop):

    printed = set()

    def __init__(self, *args, **kwargs):
        super(ParLoop, self).__init__(*args, **kwargs)
        self.kernel.cpp = True

    def prepare_arglist(self, iterset, *args):
        nbytes = 0

        arglist = iterset._kernel_args_
        for arg in args:
            arglist += arg._kernel_args_
            if arg.access is base.INC:
                nbytes += arg.data.nbytes * 2
            else:
                nbytes += arg.data.nbytes
        seen = set()
        for arg in args:
            maps = arg.map_tuple
            for map_ in maps:
                for k in map_._kernel_args_:
                    if k in seen:
                        continue
                    arglist += map_._kernel_args_
                    seen.add(k)
                    nbytes += map_.values.nbytes

        self.nbytes = nbytes

        return arglist

    @collective
    def reduction_end(self):
        """End reductions"""
        if not self._has_reduction:
            return
        with self._reduction_event_end:
            for arg in self.global_reduction_args:
                arg.reduction_end(self.comm)
            # Finalise global increments
            for tmp, glob in self._reduced_globals.items():
                # copy results to the host
                raise NotImplementedError()
                cuda.memcpy_dtoh(tmp._data, tmp.device_handle)
                glob._data += tmp._data

    @cached_property
    def _jitmodule(self):
        return JITModule(self.kernel, self.iterset, *self.args,
                         iterate=self.iteration_region,
                         pass_layer_arg=self._pass_layer_arg,
                         delay=True)

    @collective
    def _compute(self, part, fun, *arglist):
        if part.size == 0:
            return

        with timed_region("ParLoop_{0}_{1}".format(self.iterset.name, self._jitmodule._wrapper_name)):
            fun(part.offset, part.offset + part.size, *arglist)


class OpenCLBackend(AbstractPETScBackend):
    ParLoop_offloading = ParLoop
    ParLoop_no_offloading = sequential.ParLoop
    ParLoop = sequential.ParLoop
    Set = base.Set
    ExtrudedSet = ExtrudedSet
    MixedSet = base.MixedSet
    Subset = Subset
    DataSet = DataSet
    MixedDataSet = petsc_base.MixedDataSet
    Map = Map
    MixedMap = base.MixedMap
    Dat = Dat
    MixedDat = petsc_base.MixedDat
    DatView = base.DatView
    Mat = petsc_base.Mat
    Global = Global
    GlobalDataSet = petsc_base.GlobalDataSet
    PETScVecType = 'viennacl'

    def __init__(self):
        self.offloading = False

    @cached_property
    def context(self):
        # create a dummy vector and extract its underlying context
        x = PETSc.Vec().create(PETSc.COMM_WORLD)
        x.setType('viennacl')
        x.setSizes(size=1)
        ctx_ptr = x.getCLContextHandle()
        return cl.Context.from_int_ptr(ctx_ptr, retain=False)

    @cached_property
    def queue(self):
        # create a dummy vector and extract its associated command queue
        x = PETSc.Vec().create(PETSc.COMM_WORLD)
        x.setType('viennacl')
        x.setSizes(size=1)
        queue_ptr = x.getCLQueueHandle()
        return cl.CommandQueue.from_int_ptr(queue_ptr, retain=False)

    def turn_on_offloading(self):
        self.offloading = True
        self.ParLoop = self.ParLoop_offloading

    def turn_off_offloading(self):
        self.offloading = False
        self.ParLoop = self.ParLoop_no_offloading


opencl_backend = OpenCLBackend()
