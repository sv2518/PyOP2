from abc import ABC, abstractmethod

from pyop2.caching import cached_property  


class Arg(ABC):

    def __init__(self, comm):
        self._comm = comm

    @property
    def comm(self):
        return self._comm


class DataCarrierArg(Arg, ABC):

    """An argument to a :func:`pyop2.op2.par_loop`.

    .. warning ::
        User code should not directly instantiate :class:`Arg`.
        Instead, use the call syntax on the :class:`DataCarrier`.
    """

    def __init__(self, comm, access, dtype, map_=None, unroll_map=False):
        """
        :param data: A data-carrying object, either :class:`Dat` or class:`Mat`
        :param map:  A :class:`Map` to access this :class:`Arg` or the default
                     if the identity map is to be used.
        :param access: An access descriptor of type :class:`Access`
        :param lgmaps: For :class:`Mat` objects, a tuple of 2-tuples of local to
            global maps used during assembly.

        Checks that:

        1. the maps used are initialized i.e. have mapping data associated, and
        2. the to Set of the map used to access it matches the Set it is
           defined on.

        A :class:`MapValueError` is raised if these conditions are not met."""

        # TODO: Remove this circular import
        from pyop2.base import Map

        super().__init__(comm)

        self._dtype = dtype
        self._map = map_
        if map_ is None:
            self.map_tuple = ()
        elif isinstance(map_, MapArg):
            self.map_tuple = (map_, )
        else:
            self.map_tuple = tuple(map_)

        if dtype.kind == "c" and (access == MIN or access == MAX):
            raise ValueError("MIN and MAX access descriptors are undefined on complex data.")
        self._access = access
        self.unroll_map = unroll_map

        # Check arguments for consistency
        # if configuration["type_check"] and not (self._is_global or map is None):
        #     for j, m in enumerate(map):
        #         if m.iterset.total_size > 0 and len(m.values_with_halo) == 0:
        #             raise MapValueError("%s is not initialized." % map)
        #         if self._is_mat and m.toset != data.sparsity.dsets[j].set:
        #             raise MapValueError(
        #                 "To set of %s doesn't match the set of %s." % (map, data))
        #     if self._is_dat and map.toset != data.dataset.set:
        #         raise MapValueError(
        #             "To set of %s doesn't match the set of %s." % (map, data))

    @property
    def _wrapper_cache_key_(self):
        return self._key

    @property
    def _key(self):
        return type(self), self._map, self._access

    def __eq__(self, other):
        r""":class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return self._key == other._key

    def __ne__(self, other):
        r""":class:`Arg`\s compare equal of they are defined on the same data,
        use the same :class:`Map` with the same index and the same access
        descriptor."""
        return not self.__eq__(other)

    def __str__(self):
        return "OP2 Arg: dat %s, map %s, access %s" % \
            (self.data_class, self._map, self._access)

    def __repr__(self):
        return "Arg(%r, %r, %r)" % \
            (self.data_class, self._map, self._access)

    @cached_property
    def name(self):
        """The generated argument name."""
        return "arg%d" % self.position

    @cached_property
    def ctype(self):
        """String representing the C type of the data in this ``Arg``."""
        return self.data.ctype

    @cached_property
    def dtype(self):
        """Numpy datatype of this Arg"""
        return self._dtype

    @cached_property
    def map(self):
        """The :class:`Map` via which the data is to be accessed."""
        return self._map

    @cached_property
    def access(self):
        """Access descriptor. One of the constants of type :class:`Access`"""
        return self._access


class DatArg(DataCarrierArg):

    def __init__(self, *args, shape, **kwargs):
        super().__init__(*args, **kwargs)

        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def is_direct(self):
        return self._map is None

    @property
    def is_indirect(self):
        return self._map is not None


class DatViewArg(DatArg):

    def __init__(self, *args, index, **kwargs):
        super().__init__(*args, **kwargs)
        self._index = index

    @property
    def index(self):
        return self._index


class GlobalArg(DataCarrierArg):

    def __init__(self, *args, dim, **kwargs):
        super().__init__(*args, **kwargs)

        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def is_reduction(self):
        return self._access in {INC, MIN, MAX}



class MatArg(DataCarrierArg):

    def __init__(self, *args, dims, lgmaps, **kwargs):
        super().__init__(*args, **kwargs)

        if lgmaps is not None:
            lgmaps = as_tuple(lgmaps)
            #assert len(lgmaps) == self.data.nblocks

        self._dims = dims
        self._lgmaps = lgmaps


class MapArg(Arg):

    def __init__(self, arity, comm, extruded, constant_layers, offset, shape, dtype):
        super().__init__(comm)
        self._arity = arity
        self._extruded = extruded
        self._constant_layers = constant_layers
        self.offset = offset
        self.shape = shape
        self.dtype=dtype

    @property
    def arity(self):
        return self._arity

    @property
    def extruded(self):
        return self._extruded

    @property
    def constant_layers(self):
        return self._constant_layers


class MixedArg(ABC):
    
    def __init__(self, args):
        self._args = args

    def __iter__(self):
        return self

    def __next__(self):
        for arg in self._args:
            yield arg

    def split(self):
        return self._args


class MixedDatArg(MixedArg):

    def __init__(self, args):
        if not all(isinstance(arg, DatArg) for arg in args):
            raise ValueError

        super().__init__(args)


class MixedMatArg(MixedArg):

    def __init__(self, args, *, shape):
        if not all(isinstance(arg, DatArg) for arg in args):
            raise ValueError

        super().__init__(args)
        self._shape = shape

    # def split(self):
    #     rows, cols = self.shape
    #     mr, mc = self.map
    #     return tuple(_make_object('Arg', data_class=Mat, self.data[i, j], (mr.split[i], mc.split[j]),
    #                               access=self._access)
                     # for i in range(rows) for j in range(cols))


class SetArg(Arg):

    def __init__(self, *args, constant_layers=False, extruded=False, subset=False):
        super().__init__(*args)

        self.constant_layers = constant_layers
        self.extruded = extruded
        self.subset = subset


