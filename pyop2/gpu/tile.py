import loopy as lp
import numpy as np
import pycuda.driver as cuda
from math import ceil, sqrt, floor
from pytools import memoize_method
from pycuda.compiler import SourceModule
from pyop2.utils import cached_property
from pytools import ImmutableRecord


# {{{ implementing the tiling transformation

class TilingConfiguration(ImmutableRecord):
    """
    Records the configuration for :func:`pyop2.gpu.tile.tiled_transform`.

    :attr ncells_per_block: Number of cells whose computation workload is to be
        given to one CUDA block.
    :attr nthreads_per_cell: Number of CUDA threads to be launched for one each
        cell in the mesh.
    :attr matvec1_row_tile_length: Number of rows in the tile of the first
        matvec (first matvec := quadrature stage)
    :attr matvec1_col_tile_length: Number of columns in the tile of the first
        matvec (first matvec := quadrature stage)
    :attr matvec2_row_tile_length: Number of rows in the tile of the second
        matvec (second matvec := output DoF stage)
    :attr matvec2_col_tile_length: Number of columns in the tile of the second
        matvec (second matvec := output DoF stage)
    :attr load_coordinates_to_shared: Should the coordinates of the cell be
        prefeteched to shared memory?
    :attr load_input_to_shared: Should the input DoFs be prefetched to shared
        memory?
    :attr load_mats_to_shared: Should the local FEM operator matrices be loaded
        to shared memory?
    :attr load_quad_weights_to_shared: Should the quadrature weigts be loaded
        to shared memory?
    :attr tiled_prefetch_of_inputs: If input DoFs are prefetched to shared
        memory, should they be prefetched in tile lengths?
    :attr tiled_prefetch_of_quad_weights: If the quadrature weights are
        prefethced to shared memory, should they in prefetched in tile lengths?
    """
    def __init__(self,
                 ncells_per_block,
                 nthreads_per_cell,
                 operator_tile_descriptions,
                 quad_rowtile_lengths,
                 load_coordinates_to_shared,
                 load_input_to_shared,
                 load_mats_to_shared,
                 load_quad_weights_to_shared,
                 tiled_prefetch_of_inputs,
                 tiled_prefetch_of_quad_weights):
        super(TilingConfiguration, self).__init__(ncells_per_block=ncells_per_block,
                                                  nthreads_per_cell=nthreads_per_cell,
                                                  operator_tile_descriptions=operator_tile_descriptions,
                                                  quad_rowtile_lengths=quad_rowtile_lengths,
                                                  load_coordinates_to_shared=load_coordinates_to_shared,
                                                  load_input_to_shared=load_input_to_shared,
                                                  load_mats_to_shared=load_mats_to_shared,
                                                  load_quad_weights_to_shared=load_quad_weights_to_shared,
                                                  tiled_prefetch_of_inputs=tiled_prefetch_of_inputs,
                                                  tiled_prefetch_of_quad_weights=tiled_prefetch_of_quad_weights)

    def stringify(self):
        optile_str = ':'.join('('+'x'.join(str(o) for o in optiles)+')'
                              for optiles in self.operator_tile_descriptions)
        quadtile_str = ':'.join(str(q) for q in
                                self.quad_rowtile_lengths) or '()'

        strng = "%d, %d, %s, %s" % (self.ncells_per_block, self.nthreads_per_cell, optile_str, quadtile_str)

        return strng


def _make_tv_array_arg(tv):
    assert tv.address_space != lp.AddressSpace.PRIVATE
    arg = lp.ArrayArg(name=tv.name,
                      dtype=tv.dtype,
                      shape=tv.shape,
                      dim_tags=tv.dim_tags,
                      offset=tv.offset,
                      dim_names=tv.dim_names,
                      order=tv.order,
                      alignment=tv.alignment,
                      address_space=tv.address_space,
                      is_output_only=not tv.read_only)
    return arg


class MatvecStageDescr(ImmutableRecord):
    def __init__(self, dof_name, row_iname, col_iname, deriv_matrices):
        assert isinstance(dof_name, str)
        assert isinstance(row_iname, str)
        assert isinstance(col_iname, str)
        assert isinstance(deriv_matrices, set)
        super(MatvecStageDescr, self).__init__(dof_name=dof_name,
                                               row_iname=row_iname,
                                               col_iname=col_iname,
                                               deriv_matrices=deriv_matrices)


class KernelMetadata(ImmutableRecord):
    def __init__(self, **kwargs):
        assert isinstance(kwargs['iquad'], str)
        assert isinstance(kwargs['coords'], str)
        assert isinstance(kwargs['outDoF_init_iname'], str)
        assert isinstance(kwargs["quad_weights"], str)
        assert isinstance(kwargs["matvec_stage_descrs"], list)
        assert isinstance(kwargs["eval_results"], frozenset)
        assert isinstance(kwargs['scatter_iname'], str)
        super(KernelMetadata, self).__init__(**kwargs)

    @property
    def outDoF(self):
        return self.matvec_stage_descrs[-1].dof_name

    def nquad(self, kernel):
        return int(lp.symbolic.pw_aff_to_expr(kernel.get_iname_bounds(self.iquad, constants_only=True).size))

    def n_outDoF(self, kernel):
        ioutdof = self.matvec_stage_descrs[-1].row_iname
        return int(lp.symbolic.pw_aff_to_expr(kernel.get_iname_bounds(ioutdof, constants_only=True).size))

    def n_trialDoFs(self, kernel):
        itrialDoFs = [mv_stg_descr.col_iname for mv_stg_descr in self.matvec_stage_descrs[:-1]]
        return [int(lp.symbolic.pw_aff_to_expr(kernel.get_iname_bounds(itrialDoF, constants_only=True).size))
                for itrialDoF in itrialDoFs]

    @property
    def n_trial(self):
        return len(self.matvec_stage_descrs) - 1


def inference_which_should_ideally_be_done_by_passing_metadata(kernel):
    """
    Only intended to work for the vanilla representation of the form kernel.
    For ex. Sum factorized action kernels won't fit the pattern.
    """
    from pymbolic.primitives import Variable

    # quad iname
    # Assumption: There is only a single iname responsible for quadrature and
    # it starts with 'form_ip'>
    iquad, = [iname for iname in kernel.all_inames() if iname.startswith('form_ip')]

    # trialDof_x_outputDofs_x_coords: A set containing the variable names for the
    # *temporaries* of trialDofs, outputDofs and the coordinates.
    # trialDof, outputDofs, coords := local DoFs
    # These are also the variables which are written (or initialized during the
    # gather phase).
    trialDofs_x_outDof_x_coords = set().union(*(insn.write_dependency_names()
                                                for insn in kernel.instructions
                                                if 'gather' in insn.tags)) - kernel.all_inames()

    # {{{ extract outputDoF name

    # Assumption: There is only one output DoF being generated in our 1-form
    # assembly.
    # In the 'quadr' phase of the form kernel the *only* variable being written
    # is output DoF
    outDoF = set()
    outDoF, = [insn.assignee_name
               for insn in kernel.instructions
               if 'quadr' in insn.tags]

    # }}}

    # {{{ extract coords name

    # Assumptions: coordinate transformation is affine i.e. one
    # Jacobian computation for each cell. Thereby all the instructions
    # responsible for computing entries of the Jacobian matrix would be only
    # within the 'n' loop.
    coords = set()
    for insn in kernel.instructions:
        if ('eval' in insn.tags) and (insn.within_inames == frozenset(["n"])):
            coords = coords | (insn.read_dependency_names() & trialDofs_x_outDof_x_coords)

    coords, = coords

    # }}}

    # {{{ extract trial DoF names

    trialDoFs = trialDofs_x_outDof_x_coords - frozenset([coords, outDoF])

    # }}}

    # {{{ scatter iname

    # Logic: Already assumed that there is only one outDof pet kernel; so
    # picking up the scatter insn based on that singleton variable.

    scatter_insn, = [insn for insn in kernel.instructions if 'scatter' in insn.tags]
    scatter_map = scatter_insn.assignee.index_tuple[0]
    scatter_iname, = set(scatter_map.index_tuple) - set([Variable('n')])
    scatter_iname = scatter_iname.name

    # }}}

    # {{{ output DoF init iname

    # Assumption the outDoF init instruction is as follows:
    # outDoF[outDoF_init_iname, ...] <- 0

    outDoF_init_iname, = [insn.assignee.index_tuple[1].name
                          for insn in kernel.instructions
                          if ('gather' in insn.tags) and (outDoF == insn.assignee_name)]

    # }}}

    # {{{ doF_inames_in_eval_stage

    # dof_inames_in_eval_stage: iname corresponding the reduction loop in the
    # eval matvecs. These inames have been represented by $i_1$, $i_2$, ... in the
    # paper.
    doF_inames_in_eval_stage = set()
    trialDofs_to_redn_inames = {}
    for trialDoF in trialDoFs:
        # trialDoF is read only in the accumulate instruction in the matvec of
        # the eval stage and the instruction accessing would it have the
        # inames: 'n, iquad, i_1'.  Over here we extract what's the name of i_1
        # in our FEM kernel.
        iname, = set().union(*(insn.within_inames
                               for insn in kernel.instructions
                               if trialDoF in insn.read_dependency_names())) - {'n', iquad}

        doF_inames_in_eval_stage.add(iname)
        trialDofs_to_redn_inames[trialDoF] = iname

    # }}}

    # {{{ tagging the stages of the kernel

    new_insns = []

    done_with_gather = False
    done_with_jacobi_eval = False
    done_with_eval_init = False
    done_with_eval_reduction = False
    done_with_eval_wrap_up = False
    done_with_quadr_reduction = False

    for insn in kernel.instructions:
        if not done_with_gather:
            if 'gather' not in insn.tags:
                done_with_gather = True
            else:
                new_insns.append(insn)
                continue
        if not done_with_jacobi_eval:
            if iquad in insn.within_inames:
                done_with_jacobi_eval = True
            else:
                new_insns.append(insn.copy(tags=insn.tags | frozenset(["jacobi"])))
                continue
        if not done_with_eval_init:
            if doF_inames_in_eval_stage & insn.within_inames:
                done_with_eval_init = True
            else:
                new_insns.append(insn.copy(tags=insn.tags | frozenset(["eval_init"])))
                continue
        if not done_with_eval_reduction:
            if doF_inames_in_eval_stage & insn.within_inames:
                new_insns.append(insn.copy(tags=insn.tags | frozenset(["eval_redn"])))
                continue
            else:
                done_with_eval_reduction = True
        if not done_with_eval_wrap_up:
            if 'quadr' in insn.tags:
                done_with_eval_wrap_up = True
            else:
                new_insns.append(insn.copy(tags=insn.tags | frozenset(["eval_wrap_up"])))
                continue
        if not done_with_quadr_reduction:
            if iquad not in insn.within_inames:
                done_with_quadr_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags | frozenset(["quadr_redn"])))
                continue

        assert 'scatter' in insn.tags
        new_insns.append(insn)

    kernel = kernel.copy(instructions=new_insns)
    assert done_with_quadr_reduction

    # }}}

    # {{{ extract deriv_matrices, quad_weights

    # derivative matrices are the constant data whose array dimensions > 1
    deriv_matrices = {tv.name
                      for tv in kernel.temporary_variables.values()
                      if tv.initializer is not None and len(tv.initializer.shape) != 1}

    # quad_weights is the only constant data in the kernel which is a single
    # dimensional array
    quad_weights, = [tv.name
                     for tv in kernel.temporary_variables.values()
                     if tv.initializer is not None and len(tv.initializer.shape) == 1]
    # }}}

    matvec_descrs = []

    # {{{ identify matvec stages for the eval-part of the compute kernel

    from loopy.match import parse_match

    for i, (trialDoF, redn_iname) in enumerate(trialDofs_to_redn_inames.items()):
        within = parse_match("writes:%s" % trialDoF)
        trialDof_init_insn_id, = [insn.id for insn in kernel.instructions if within(kernel, insn)]

        # all the recursive reverse dependencies of trialDoF_init_insn in the
        # eval-part of the kernel form the trialDoF's matvec

        from loopy.kernel.tools import find_recursive_reverse_dependencies
        matvec_insn_ids = find_recursive_reverse_dependencies(kernel, {trialDof_init_insn_id})
        kernel = lp.tag_instructions(kernel, 'matvec%d' % i, '(' + ' or '.join(['id:%s' % matvec_insn_id
                                                                                for matvec_insn_id in matvec_insn_ids]) + ') and tag:eval')
        vars_written_in_matvec = set().union(*(insn.write_dependency_names()
                                               for insn in kernel.instructions
                                               if 'matvec%d' % i in insn.tags))
        eval_init_insn_ids = [insn.id
                              for insn in kernel.instructions
                              if (insn.assignee_name in vars_written_in_matvec) and 'eval_init' in insn.tags]

        kernel = lp.tag_instructions(kernel,
                                     'matvec%d' % i,
                                     ' or '.join(['id:%s' % eval_init_insn_id
                                                  for eval_init_insn_id in eval_init_insn_ids]))

        deriv_matrices_in_current_mv_stg = set().union(*(insn.read_dependency_names()
                                                         for insn in kernel.instructions
                                                         if 'matvec%d' % i in insn.tags)) & deriv_matrices

        matvec_descrs.append(MatvecStageDescr(trialDoF, iquad, redn_iname, deriv_matrices_in_current_mv_stg))

    # }}}

    # {{{ extract matvec producing outDoF

    (quadr_stage_DoF_iname,), = {(insn.within_inames - {'n', iquad})
                                 for insn in kernel.instructions
                                 if 'quadr' in insn.tags}

    kernel = lp.tag_instructions(kernel,
                                 'matvec%d' % (i+1),
                                 'reads:{0} or writes:{0}'.format(outDoF))
    kernel = lp.tag_instructions(kernel, 'quadr_init', 'tag:gather and'
                                 ' tag:matvec%d' % (i+1))
    kernel = lp.tag_instructions(kernel, 'quadr_wrap_up', 'tag:scatter and'
                                 ' tag:matvec%d' % (i+1))

    deriv_matrices_in_current_mv_stg = set().union(*(insn.read_dependency_names()
                                                     for insn in kernel.instructions
                                                     if 'matvec%d' % (i+1) in insn.tags)) & deriv_matrices

    matvec_descrs.append(MatvecStageDescr(outDoF, quadr_stage_DoF_iname, iquad, deriv_matrices_in_current_mv_stg))

    # }}}

    # eval_results: temporary variables which are the final result of the
    # evaluation part of the kernel.
    # Hence, eval_results =Variables which are written in the eval stage and
    # read in the quadr stage
    eval_results = (frozenset().union(*[insn.write_dependency_names()
                                        for insn in kernel.instructions
                                        if 'eval_wrap_up' in insn.tags])
                    & set().union(*[insn.read_dependency_names()
                                    for insn in kernel.instructions
                                    if 'quadr' in insn.tags]))

    return kernel, KernelMetadata(iquad=iquad,
                                  coords=coords,
                                  outDoF_init_iname=outDoF_init_iname,
                                  quad_weights=quad_weights,
                                  matvec_stage_descrs=matvec_descrs,
                                  scatter_iname=scatter_iname,
                                  eval_results=eval_results)


def tiled_transform(kernel, callables_table, tiling_config):
    """
    :param tiling_config: An instance of :class:`pyop2.gpu.tiling_config
    """

    # kernel = lp.assume(kernel, "end=%d" % (512*512*2))

    assert isinstance(tiling_config, TilingConfiguration)

    # {{{ remove noops

    noop_insns = set([insn.id
                      for insn in kernel.instructions
                      if isinstance(insn, lp.NoOpInstruction)])
    kernel = lp.remove_instructions(kernel, noop_insns)

    from loopy.transform.instruction import remove_unnecessary_deps
    kernel = remove_unnecessary_deps(kernel)

    # }}}

    # {{{ Inferring variables

    kernel, metadata = inference_which_should_ideally_be_done_by_passing_metadata(kernel)
    iquad = metadata.iquad
    coords = metadata.coords
    outDoF = metadata.outDoF
    outDoF_init_iname = metadata.outDoF_init_iname
    scatter_iname = metadata.scatter_iname
    quad_weights = metadata.quad_weights
    matvec_stage_descrs = metadata.matvec_stage_descrs
    eval_results = metadata.eval_results
    nquad = metadata.nquad(kernel)
    n_outDoF = metadata.n_outDoF(kernel)
    n_trialDoFs = metadata.n_trialDoFs(kernel)
    n_trial = metadata.n_trial

    # }}}

    nc = tiling_config.ncells_per_block
    nt = tiling_config.nthreads_per_cell
    mv_tiles = tiling_config.operator_tile_descriptions
    q_tile_lens = tiling_config.quad_rowtile_lengths[:]

    if mv_tiles == ():
        mv_tiles = tuple((nquad, nDoF) for nDoF in n_trialDoFs) + ((n_outDoF, nquad),)

    if q_tile_lens == ():
        q_tile_lens = (nquad,)

    assert all(len(tile) == 2 for tile in mv_tiles)
    assert len(mv_tiles) == len(matvec_stage_descrs)  # one for each mv stage
    assert len({op_tile_descr[0] for op_tile_descr in mv_tiles[:-1]}) == 1  # in the general case only one $T_e^r$ is supported
    assert len(q_tile_lens) == 1  # in every action kernel there is a single quadrature loop

    T_e_r = mv_tiles[0][0]
    T_e_cs = [tile[1] for tile in mv_tiles[:-1]]
    T_q_r = mv_tiles[-1][0]
    T_q_c = mv_tiles[-1][1]

    kernel = lp.split_iname(kernel, iquad, q_tile_lens[0], outer_iname='iquad_tile')
    kernel = lp.rename_iname(kernel, iquad+"_inner", iquad)

    # {{{ privatize temps for function evals and make them LOCAL

    kernel = lp.privatize_temporaries_with_inames(kernel, iquad, eval_results)

    # FIXME: Corner case: q_tile_lens == (1,). This would be the SCPT.
    kernel = lp.set_temporary_scope(kernel, eval_results, lp.AddressSpace.LOCAL)

    # }}}

    # {{{ cast scalars which occur in both the matvecs as substs

    from loopy.match import parse_match

    within = parse_match('tag:eval_wrap_up and not tag:matvec*')
    proxy_quad_wt_scalars = [insn.assignee.name for insn in kernel.instructions if within(kernel, insn)]

    for proxy_quad_wt_scalar in proxy_quad_wt_scalars:
        assert kernel.temporary_variables[proxy_quad_wt_scalar].shape == ()
        kernel = lp.assignment_to_subst(kernel, proxy_quad_wt_scalar, extra_arguments=(iquad,))

    assert len(kernel.substitutions) == len(proxy_quad_wt_scalars)
    kernel = lp.expand_subst(kernel)

    # }}}

    # {{{ Duplicate inames to separate transformation logic for different matvecs

    for i, mv_stg_descr in enumerate(matvec_stage_descrs):
        kernel = lp.duplicate_inames(kernel, mv_stg_descr.col_iname, "tag:matvec%d" % i, "icol%d" % i)

    kernel = lp.duplicate_inames(kernel, iquad, "tag:eval", "irow_eval")
    kernel = lp.duplicate_inames(kernel, matvec_stage_descrs[-1].row_iname, "tag:quadr", "irow_quadr")

    # }}}

    # {{{ change address space of constants to '__global'

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [tv.initializer.flatten() for tv in old_temps.values() if tv.initializer is not None]

    new_temps = dict((tv.name, tv) for tv in old_temps.values() if tv.initializer is None)
    kernel = kernel.copy(args=kernel.args+[_make_tv_array_arg(tv)
                                           for tv in old_temps.values()
                                           if tv.initializer is not None],
                         temporary_variables=new_temps)

    # }}}

    from loopy.loop import fuse_loop_domains
    kernel = fuse_loop_domains(kernel)

    from loopy.transform.data import remove_unused_axes_in_temporaries
    kernel = remove_unused_axes_in_temporaries(kernel)

    # Realize CUDA blocks
    kernel = lp.split_iname(kernel, "n", nc, outer_iname="iblock", inner_iname="icell")

    # Privatize eval_results
    kernel = lp.privatize_temporaries_with_inames(kernel, 'icell', only_var_names=eval_results)

    # cut down the size of the number of basis coeffs written by each
    # thread(if there are multiple threads)
    kernel = lp.rename_iname(kernel, scatter_iname, "irow_quadr", True)
    kernel = lp.rename_iname(kernel, outDoF_init_iname, "irow_quadr", True)

    from loopy.transform.make_scalar import remove_axis
    kernel = remove_axis(kernel, outDoF, 0)

    # enfoce dependency of first matvec stage onto the jacobian evaluation stage
    kernel = lp.add_dependency(kernel, 'tag:eval_init and tag:matvec0', 'tag:jacobi')

    for i in range(n_trial):
        kernel = lp.add_dependency(kernel,
                                   '(tag:eval_init or tag:quadr_init) and tag:matvec%d' % (i+1),
                                   'tag:eval_wrap_up and tag:matvec%d' % i)

    if tiling_config.load_coordinates_to_shared:
        # FIXME: This configuration parameter seems unnecessary as of now. I
        # might choose not to support it.
        kernel = lp.privatize_temporaries_with_inames(kernel, 'icell', [coords])
        kernel = lp.assignment_to_subst(kernel, coords)
        raise NotImplementedError("This might be only useful for high order meshes.")

    # Splitting row in eval stage
    kernel = lp.split_iname(kernel, "irow_eval", T_e_r, outer_iname="irowtile_eval")
    # Splitting column in eval stage
    for i, T_e_c in enumerate(T_e_cs):
        kernel = lp.split_iname(kernel, "icol%d" % i, T_e_c, outer_iname='icoltile%d' % i)

    # Splitting row in the quadr stage
    kernel = lp.split_iname(kernel, "irow_quadr", T_q_r, inner_iname="irow%d_inner" % n_trial, outer_iname="irowtile_quadr")
    # Splitting column in quadr stage
    kernel = lp.split_iname(kernel, "icol%d" % n_trial, T_q_c, outer_iname="icoltile%d" % n_trial)

    # decoupling the tile's inner-products of separate trial functions
    for i in range(n_trial):
        kernel = lp.rename_iname(kernel, "irow_eval_inner", "irow%d_inner" % i, within="tag:matvec%d" % i)

    # {{{ Prefetch inputDoFs (not implemented)

    if tiling_config.load_input_to_shared:
        raise NotImplementedError("More like NotYetImplementedError.")
        # kernel = lp.privatize_temporaries_with_inames(kernel, 'icell',
        #         only_var_names=inputDoFs)
        # from loopy.transform.precompute import precompute_for_single_kernel
        # for i, inputDoF in enumerate(inputDoFs):
        #     kernel = lp.assignment_to_subst(kernel, inputDoF)
        #     input_prcmpt_iname = 'input_basis_prcmpt'
        #     if tiling_config.tiled_prefetch_of_inputs:
        #         sweep_inames = (doF_inames_in_quad_stage[i]+'_inner', 'icell')
        #         outer_inames = 'iblock,icoltile_matvec1,irowtile_matvec1'
        #     else:
        #         sweep_inames = ('icoltile_matvec1', doF_inames_in_quad_stage[i]+'_inner', 'icell')
        #         outer_inames = 'iblock'
        #     kernel = precompute_for_single_kernel(kernel, callables_table,
        #             subst_use=doF_inames_in_quad_stage[i]+'_subst',
        #             sweep_inames=sweep_inames,
        #             precompute_outer_inames=outer_inames,
        #             precompute_inames=(input_prcmpt_iname, 'icell'),
        #             temporary_address_space=lp.AddressSpace.LOCAL,
        #             default_tag=None,
        #             )
        #     kernel = lp.split_iname(kernel, input_prcmpt_iname,
        #             nt, inner_tag="l.0")

    # }}}

    # {{{ Prefetch deriv matrices

    total_shared_vars = []

    if tiling_config.load_mats_to_shared:
        from loopy.transform.data import add_prefetch_for_single_kernel

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()

        for istage, mv_stg_descr in enumerate(matvec_stage_descrs):
            if istage < n_trial:
                # eval stage
                fetch_outer_inames = 'iblock,iquad_tile,icoltile{0},irowtile_eval'.format(istage)
                tr = T_e_r
                tc = T_e_cs[istage]
            else:
                # quadr stage
                fetch_outer_inames = 'iblock,iquad_tile,icoltile{0},irowtile_quadr'.format(istage)
                tr = T_q_r
                tc = T_q_c

            # sweep the row, column of the tile.
            sweep_inames = "irow{0}_inner, icol{0}_inner".format(istage)
            prefetch_inames = [vng("iprftch") for _ in range(2)]

            # prefetch all the derivative matrices in the current matvec stage
            for i_op_pos, prftch_from in enumerate(mv_stg_descr.deriv_matrices):
                prftch_into = vng('matvec%d_cnst_mtrix_prftch' % istage)
                total_shared_vars.append(prftch_into)

                kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                                                        var_name=prftch_from,
                                                        sweep_inames=sweep_inames,
                                                        temporary_address_space=lp.AddressSpace.LOCAL,
                                                        dim_arg_names=prefetch_inames,
                                                        temporary_name=prftch_into,
                                                        compute_insn_id=ing("prftch_matvec%d" % istage),
                                                        fetch_outer_inames=fetch_outer_inames,
                                                        default_tag=None,
                                                        within='tag:matvec%d' % istage)

                new_temps = kernel.temporary_variables.copy()

                lx, ly = kernel.temporary_variables[prftch_into].shape
                assert lx*ly == tr*tc
                # prefetch the matrices into a single shared memory location
                # with the appropriate offsets
                new_temps[prftch_into] = (kernel.temporary_variables[prftch_into].copy(base_storage='prftch_matrix_base',
                                                                                       offset=i_op_pos*tr*tc,
                                                                                       shape=((i_op_pos+1)*lx, ly)))

                kernel = kernel.copy(temporary_variables=new_temps)

            # add a dependency of a (i+1)-th matvec stage on the prefetch
            # instructions of ith matvec stage
            kernel = lp.add_dependency(kernel,
                                       'tag:matvec%d and (tag:eval_redn or tag:quadr_redn)' % istage,
                                       'id:prftch_matvec%d*' % istage)
            kernel = lp.add_nosync(kernel, source='id:prftch_matvec%d*' % istage,
                                   sink='id:prftch_matvec%d*' % istage,
                                   scope='local', empty_ok=True, force=True)

            # join inames to promote more coalesced memory accesses in the
            # prefetches
            kernel = lp.join_inames(kernel, prefetch_inames, new_iname='i_matvec%d_prftch' % istage)
            kernel = lp.split_iname(kernel, 'i_matvec%d_prftch' % istage, nc*nt)  # , outer_tag="ilp")
            kernel = lp.split_iname(kernel,
                                    'i_matvec%d_prftch_inner' % istage,
                                    nt, inner_tag='l.0', outer_tag='l.1')

        # {{{ prefetch of (i+1)-th matvec stage should depend on prefetch of
        # (i)th matvec stage

        for i in range(n_trial):
            kernel = lp.add_dependency(kernel, 'id:prftch_matvec%d*' % (i+1),
                                       'tag:eval_wrap_up and tag:matvec%d' % i)

        # }}}

    # }}}

    # {{{ Prefetch: Quad Weights

    if tiling_config.load_quad_weights_to_shared:
        # FIXME: instead of prefetching this we should precompute the constant
        # term which we made as a substitution.
        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()
        quad_weight_prefetch_insn = ing("quad_wt_prftch_insn")
        quad_weight_prefetch_iname = vng("iprtftch")

        if tiling_config.tiled_prefetch_of_quad_weights:
            raise NotImplementedError("Not sure if this is any fruitful!")
        else:
            sweep_inames = ['iquad_tile', 'irowtile_eval'] + ['irow%d_inner' % i
                                                              for i in range(n_trial)]
            fetch_outer_inames = 'iblock'

        from loopy.transform.data import add_prefetch_for_single_kernel
        kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                                                var_name=quad_weights,
                                                sweep_inames=sweep_inames,
                                                temporary_address_space=lp.AddressSpace.LOCAL,
                                                dim_arg_names=(quad_weight_prefetch_iname,),
                                                temporary_name='cnst_quad_weight_prftch',
                                                compute_insn_id=quad_weight_prefetch_insn,
                                                fetch_outer_inames=fetch_outer_inames,
                                                default_tag=None)

        kernel = lp.add_dependency(kernel, "tag:matvec0 and tag:eval_init", "id:%s" %
                                   quad_weight_prefetch_insn)

        kernel = lp.split_iname(kernel, quad_weight_prefetch_iname, nc * nt)  # , outer_tag="ilp")
        kernel = lp.split_iname(kernel, quad_weight_prefetch_iname+'_inner', nt,
                                outer_tag="l.1", inner_tag="l.0")

    # }}}

    # {{{ divide the matvec of each cell across threads

    for i in range(n_trial+1):
        kernel = lp.split_iname(kernel, "irow%d_inner" % i, nt, inner_tag="l.0")  # , outer_tag="ilp")

    # }}}

    # privatizing accumulators (necessary for logic preservation)
    for i, (redn_length, tc) in enumerate(zip(n_trialDoFs+[nquad], T_e_cs+[T_q_c])):
        if tc < redn_length:
            from loopy.match import parse_match
            redn_accumulators = [insn.assignee_name
                                 for insn in kernel.instructions
                                 if parse_match("(tag:eval_init or tag:quadr_init) and"
                                                " tag:matvec%d" % i)(kernel, insn)]

            kernel = lp.privatize_temporaries_with_inames(kernel, 'irow%d_inner_outer' % i,
                                                          only_var_names=redn_accumulators)
            kernel = lp.duplicate_inames(kernel, ['irow%d_inner_outer' % i],
                                         within='tag:matvec%d and (tag:eval_wrap_up or tag:quadr_wrap_up)' % i)
            kernel = lp.duplicate_inames(kernel, ['irow%d_inner_outer' % i],
                                         within='tag:matvec%d and (tag:eval_init or tag:quadr_init)' % i)
        else:
            # if tc == redn_length: number of column tiles is zero and
            # privatization can be avoided by promoting all the instructions in
            # the matvec stage into icoltile.
            kernel = lp.add_inames_to_insn(kernel, 'icoltile%d' % i,
                                           'tag:matvec%d' % i)

    if q_tile_lens[0] != nquad:
        # essentially privatize irow_tileblahblah for outputDoF.
        kernel = lp.privatize_temporaries_with_inames(kernel,
                                                      'irowtile_quadr',
                                                      only_var_names=[outDoF])
        kernel = lp.duplicate_inames(kernel, 'irowtile_quadr',
                                     within='tag:quadr_wrap_up')
        kernel = lp.duplicate_inames(kernel, 'irowtile_quadr',
                                     within='tag:quadr_init')
        kernel = lp.remove_dependency(kernel,
                                      'tag:quadr_init',
                                      'tag:eval_wrap_up')
    else:
        kernel = lp.add_inames_to_insn(kernel, 'iquad_tile', 'tag:matvec*')

    kernel = lp.tag_inames(kernel, "icell:l.1, iblock:g.0")

    kernel = lp.remove_unused_inames(kernel)
    kernel = kernel.copy(loop_priority=frozenset())

    return kernel, args_to_make_global

# }}}


# {{{ auto tile

WARP_SIZE = 32


class AutoTiler:
    """
    Helper class to tune the :class:`pyop2.gpu.tile.TilingConfiguration` for
    :func:`pyop2.gpu.tile.tiled_transform`. Tuning heuristic applied as
    specified in Paper xx. All the mathematical symbols used in the docs of the
    member methods are defined the paper.

    :attr fem_program: An instance of :class:`loopy.program.Program` which is
        the FEM computational kernel to be tuned.

    See the entrypoint :func:`pyop2.gpu.tile.Autotiler.__call__`
    """
    def __init__(self, fem_program, num_candidate_knls):
        self.fem_program = fem_program
        self.num_candidate_knls = num_candidate_knls

    @cached_property
    def nbasis(self):
        return int(lp.symbolic.pw_aff_to_expr(self.fem_program.root_kernel.get_iname_bounds('form_i',
                                                                                            constants_only=True).size))

    @cached_property
    def nquad(self):
        r"""
        Returns $N_{\mathrm{quad}}$.
        """
        return int(lp.symbolic.pw_aff_to_expr(self.fem_program.root_kernel.get_iname_bounds('form_ip',
                                                                                            constants_only=True).size))

    @property
    def nTrial(self):
        r"""
        Returns $N_{\mathrm{trial}}$.
        """
        raise NotImplementedError()

    @property
    def npMats(self):
        raise NotImplementedError()

    @property
    def nqMats(self):
        raise NotImplementedError()

    @property
    def nTrialDofs(self):
        r"""
        Returns a list of ($N_{\mathrm{trialDof, i}$) for i in [0,
        $N_\mathrm{trial})$.
        """
        raise NotImplementedError()

    @property
    def nOutDof(self):
        """
        Returns $N_{\mathrm{outDoF}}$.
        """
        raise NotImplementedError()
        pass

    @cached_property
    def num_const_matrices(self):
        """
        Returns the number of constant matrices in the FEM kernel.
        """
        const_matrices_in_quad = set()
        const_matrices_in_basis = set()
        const_matrices = frozenset([tv.name
                                    for tv in self.fem_program.root_kernel.temporary_variables.values()
                                    if tv.initializer is not None and len(tv.initializer.shape) == 2])

        for insn in self.fem_program.root_kernel.instructions:
            if 'quadrature' in insn.tags:
                const_matrices_in_quad.update(insn.read_dependency_names() & const_matrices)
            if 'basis' in insn.tags:
                const_matrices_in_basis.update(insn.read_dependency_names() & const_matrices)

        return max(len(const_matrices_in_quad), len(const_matrices_in_basis))

    @cached_property
    def num_func_eval_vars(self):
        """
        Returns the number of variables evaluated at the quadrature nodes.
        """
        evaluation_variables = (set().union(*[insn.write_dependency_names()
                                              for insn in self.fem_program.root_kernel.instructions
                                              if 'quadrature' in insn.tags])
                                & set().union(*[insn.read_dependency_names()
                                                for insn in self.fem_program.root_kernel.instructions
                                                if 'basis' in insn.tags]))

        return len(evaluation_variables)

    def get_nsync(self, tile_descrs):
        """
        Returns the number of block level synchronization instructions in a
        single kernel execution.
        """
        (tp_r, tp_c), (tq_r, tq_c) = tile_descrs
        return (ceil(self.nquad/tp_r) * ceil(self.nbasis/tp_c)
                + ceil(self.nbasis/tq_r) * ceil(self.nquad/tq_c))


    def get_shared_mem_allocated(self, tile_descrs):

        pass


    def theoretical_warps_per_sm(self, tiling_config):
        """
        Returns the number of warps residing on an Streaming Multiprocessor.
        """

        cells_per_block = tiling_config.ncells_per_block
        threads_per_cell = tiling_config.nthreads_per_cell
        (t1_r, t1_c), (t2_r, t2_c) = tiling_config.operator_tile_descriptions

        # {{{ computing shared mem usage per block

        shared_usage = (self.num_const_matrices*max(t1_r*t1_c, t2_r*t2_c)
                        + self.nquad
                        + self.num_func_eval_vars*self.nquad*cells_per_block)

        # convert doubles to KB
        shared_usage *= 8e-3

        # }}}

        warps_per_block = floor((threads_per_cell*cells_per_block)/32)
        blocks_per_sm = min(96//shared_usage if shared_usage < 48 else 0, 32)
        warps_per_sm = blocks_per_sm*warps_per_block

        return warps_per_sm

    def get_work_efficiency(self, tiling_config):
        """
        Returns the efficieny(as a fraction) for a tile defined by t1_r x t1_c,
        t2_r x t2_c.

        One reason for inefficiency is if the number of threads in a CUDA block
        aren't a multiple of the warp size.
        """
        cells_per_block = tiling_config.ncells_per_block
        threads_per_cell = tiling_config.nthreads_per_cell
        (t1_r, t1_c), (t2_r, t2_c) = tiling_config.operator_tile_descriptions

        # wasted work in the function evaluation stage
        wasted_work = self.nbasis*((t1_r % threads_per_cell)*(self.nquad//t1_r)
                                   + ((self.nquad % t1_r) % threads_per_cell))

        wasted_work += self.nquad*((t2_r % threads_per_cell)*(self.nbasis//t2_r)
                                   + ((self.nbasis % t2_r) % threads_per_cell))

        wasted_work_fraction = wasted_work / (2*self.nquad*self.nbasis)

        threads_in_block = threads_per_cell * cells_per_block
        warp_mismatch_factor = threads_in_block / (threads_in_block + (WARP_SIZE - (threads_in_block % WARP_SIZE)))

        return warp_mismatch_factor*(1-wasted_work_fraction)

    def actual_warps_per_sm(self, tiling_config):
        """
        Returns "actual warps residing per SM" = Efficiency * "theoretical
        warps reising per SM".
        """
        return (self.theoretical_warps_per_sm(tiling_config)
                * self.get_work_efficiency(tiling_config))

    @memoize_method
    def estimated_exec_time(self, tiling_config):
        """
        Returns a metric proportional to the execution time for a
        configuration.
        """

        n_c = tiling_config.ncells_per_block
        n_t = tiling_config.nthreads_per_cell
        (t1_r, t1_c), (t2_r, t2_c) = tiling_config.operator_tile_descriptions
        n_w = self.actual_warps_per_sm(tiling_config)

        if n_w == 0:
            return float("inf")
        n_lb = self.get_local_barriers(tiling_config.operator_tile_descriptions,
                                       tiling_config.quad_rowtile_lengths)
        n_blocks = (n_w * 32)/(n_t*n_c)

        # nb, nq = self.nbasis, self.nquad
        # return (n_t*nb + nb*nq/(n_t*n_c) + nb*nq*(n_t+n_c)/20.0)/n_w
        # return n_lb/n_blocks
        return 4.0/n_w + n_lb/n_blocks + 0.25*n_t

    def get_candiate_configs(self):

        print('Started with the whole process')

        threads_to_cells = {}

        def eta_simd(nc, nt):
            return (nc*nt) / (32.0*ceil(nc*nt/32))

        for nc in range(1, 70):
            for nt in range(1, 20):
                if eta_simd(nc, nt) > 0.97 and (nc*nt <= 256):
                    if nt in threads_to_cells:
                        threads_to_cells[nt].append(nc)
                    else:
                        threads_to_cells[nt] = [nc]

        tiles = []

        for i in range(1, ceil(sqrt(self.nbasis))+1):
            t1_c = ceil(self.nbasis/i)
            for j in range(1, ceil(sqrt(self.nquad))+1):
                t1_r = ceil(self.nquad/j)
                for k in range(1, ceil(sqrt(self.nbasis))+1):
                    t2_r = ceil(self.nbasis/k)
                    for l in range(1, ceil(sqrt(self.nquad))+1):
                        t2_c = ceil(self.nquad/l)
                        if abs(t1_r*t1_c-t2_r*t2_c)/max(t1_r*t1_c, t2_c*t2_r) < 0.2:
                            tiles.append(((t1_r, t1_c), (t2_r, t2_c)))

        params = []

        for tile in tiles:
            for threads in threads_to_cells:
                for cells in threads_to_cells[threads]:
                    params.append(TilingConfiguration(cells, threads, tile,
                                  (), False, False, True, True, False, False))

        # sort the parameters with highest occupancy.
        params.sort(key=lambda P: self.estimated_exec_time(P))

        print('Done with the thing...')

        return params[:self.num_candidate_knls]

    @memoize_method
    def convert_numpy_arrays_to_cuda_mems(self, ary):
        ary = np.array(ary)
        ary_gpu = cuda.mem_alloc(ary.nbytes)
        cuda.memcpy_htod(src=ary, dest=ary_gpu)
        return ary_gpu

    def __call__(self, args, argshapes):
        best_performing_time = float("inf")
        best_performing_config = None
        nminrounds = 15
        nwarmup = 5
        mintime = 0.1

        copied_args = ()
        for i, lpy_arg in enumerate(self.fem_program.args):
            if lpy_arg.name in self.fem_program.root_kernel.get_written_variables():
                # arg is written during kernel execution => make a copy
                arg_gpu = cuda.mem_alloc(int(np.prod(argshapes[i])*lpy_arg.dtype.itemsize))
                copied_args += (arg_gpu,)
            else:
                # arg is read only => pass the same arg to the knl
                copied_args += (args[i],)

        from pyop2.gpu.tile import tiled_transform

        for tiling_config in self.get_candiate_configs():

            for lpy_arg, arg, copied_arg, argshape in zip(self.fem_program.args, args, copied_args, argshapes):
                if lpy_arg.name in self.fem_program.root_kernel.get_written_variables():
                    # arg is written during kernel execution => make a copy
                    cuda.memcpy_dtod(src=arg, dest=copied_arg,
                                     size=int(np.prod(argshape)*lpy_arg.dtype.itemsize))

            print(75*'=')
            print('Params:', tiling_config.stringify())

            kernel, extra_args = tiled_transform(self.fem_program.root_kernel,
                                                 self.fem_program.callables_table,
                                                 tiling_config)
            # kernel = lp.assume(kernel, "end=%d" % args[1])
            from pymbolic import evaluate
            kernel = self.fem_program.with_root_kernel(kernel)
            code = lp.generate_code_v2(kernel).device_code()
            glens, llens = kernel.get_grid_size_upper_bounds_as_exprs()
            grid = tuple(int(evaluate(glens[i], {"start": args[0], "end": args[1]})) if i < len(glens) else 1
                         for i in range(2))
            block = tuple(int(evaluate(llens[i], {"start": args[0], "end": args[1]})) if i < len(llens) else 1
                          for i in range(3))

            executable_knl = SourceModule(code, options=["-use_fast_math", "-w"]).get_function(kernel.name)
            executable_knl.prepare("i"*2+"P"*len(args[2:])+"P"*len(extra_args))
            extra_args = tuple(self.convert_numpy_arrays_to_cuda_mems(tuple(arg))
                               for arg in extra_args)

            for i in range(nwarmup):
                executable_knl.prepared_call(grid, block, *(copied_args+extra_args))

            runtimes = []

            # execute the kernel for a minimum of 'nminrounds' of non-warmup
            # runs and such that it run for at least 0.1s
            while (len(runtimes) < nminrounds) or sum(runtimes) < mintime:
                start_evt = cuda.Event()
                end_evt = cuda.Event()
                start_evt.record()
                # start_evt.synchronize()
                executable_knl.prepared_call(grid, block, *(copied_args+extra_args))
                end_evt.record()
                end_evt.synchronize()
                runtimes.append(start_evt.time_till(end_evt)/1000)

            exec_time = np.average(runtimes)
            from pyop2.configuration import configuration
            print("GFlops/s = {}".format(configuration["flop_count"]/exec_time))
            print(75*'=')

            with open(configuration["output_file"], 'a') as f:
                f.write(tiling_config.stringify() + ", %.1f, %s\n" % (configuration["flop_count"]/exec_time,
                                                                      configuration["compute_tag"]))

            if exec_time < best_performing_time:
                best_performing_time = exec_time
                best_performing_config = tiling_config

        return tiled_transform(self.fem_program.root_kernel, self.fem_program.callables_table,
                               best_performing_config)

# }}}

# vim: fdm=marker
