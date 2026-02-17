"""
PyMisha - Python wrapper for the misha Genomic Data Analysis Toolkit
"""

__version__ = '0.1.5'

# Make numpy available for expressions
import numpy as np  # noqa: F401

from . import _shared
from ._shared import (
    CONFIG,
    _bound_colname,
    _checkroot,
    _chunk_slices,
    _df2pymisha,
    _iterated_intervals,
    _itr2pymisha,
    _make_progress_callback,
    _progress_context,
    _pymisha,
    _pymisha2df,
)
from .analysis import gcis_decay, gsegment, gwilcox
from .dataset import (
    gdataset_info,
    gdataset_load,
    gdataset_ls,
    gdataset_save,
    gdataset_unload,
)
from .db import (
    gdb_export_fasta,
    gdb_examples_path,
    gdb_info,
    gdb_init,
    gdb_init_examples,
    gdb_reload,
    gdb_unload,
    gsetroot,
)
from .db_attrs import gdb_get_readonly_attrs, gdb_set_readonly_attrs
from .db_create import gdb_convert_to_indexed, gdb_create, gdb_create_genome, gdb_create_linked
from .extract import gextract, gscreen
from .gdir import (
    gdir_cd,
    gdir_create,
    gdir_cwd,
    gdir_rm,
    gtrack_create_dirs,
)
from .gsynth import (
    GsynthModel,
    gsynth_bin_map,
    gsynth_load,
    gsynth_random,
    gsynth_replace_kmer,
    gsynth_sample,
    gsynth_save,
    gsynth_train,
)
from .intervals import (
    gintervals,
    gintervals_2d,
    gintervals_2d_all,
    gintervals_2d_band_intersect,
    gintervals_2d_convert_to_indexed,
    gintervals_all,
    gintervals_annotate,
    gintervals_canonic,
    gintervals_chrom_sizes,
    gintervals_convert_to_indexed,
    gintervals_coverage_fraction,
    gintervals_covered_bp,
    gintervals_dataset,
    gintervals_diff,
    gintervals_exists,
    gintervals_force_range,
    gintervals_from_bed,
    gintervals_from_strings,
    gintervals_from_tuples,
    gintervals_import_genes,
    gintervals_intersect,
    gintervals_is_bigset,
    gintervals_is_indexed,
    gintervals_load,
    gintervals_ls,
    gintervals_mapply,
    gintervals_mark_overlaps,
    gintervals_neighbors,
    gintervals_neighbors_directional,
    gintervals_neighbors_downstream,
    gintervals_neighbors_upstream,
    gintervals_normalize,
    gintervals_random,
    gintervals_rbind,
    gintervals_rm,
    gintervals_save,
    gintervals_union,
    gintervals_update,
    gintervals_window,
    giterator_cartesian_grid,
    giterator_intervals,
)
from .liftover import (
    gintervals_as_chain,
    gintervals_liftover,
    gintervals_load_chain,
    gtrack_liftover,
)
from .lookup import glookup, gtrack_lookup
from .sequence import (
    gseq_comp,
    gseq_extract,
    gseq_kmer,
    gseq_kmer_dist,
    gseq_pwm,
    gseq_rev,
    gseq_revcomp,
)
from .summary import (
    gbins_quantiles,
    gbins_summary,
    gcor,
    gdist,
    gintervals_quantiles,
    gintervals_summary,
    gpartition,
    gquantiles,
    gsample,
    gsummary,
)
from .tracks import (
    gtrack_2d_create,
    gtrack_2d_import,
    gtrack_2d_import_contacts,
    gtrack_attr_export,
    gtrack_attr_get,
    gtrack_attr_import,
    gtrack_attr_set,
    gtrack_convert_to_indexed,
    gtrack_copy,
    gtrack_create,
    gtrack_create_dense,
    gtrack_create_empty_indexed,
    gtrack_create_pwm_energy,
    gtrack_create_sparse,
    gtrack_dataset,
    gtrack_exists,
    gtrack_import,
    gtrack_import_mappedseq,
    gtrack_import_set,
    gtrack_info,
    gtrack_ls,
    gtrack_modify,
    gtrack_mv,
    gtrack_rm,
    gtrack_smooth,
    gtrack_var_get,
    gtrack_var_ls,
    gtrack_var_rm,
    gtrack_var_set,
)
from .vtracks import (
    gvtrack_clear,
    gvtrack_create,
    gvtrack_filter,
    gvtrack_info,
    gvtrack_iterator,
    gvtrack_iterator_2d,
    gvtrack_ls,
    gvtrack_rm,
)


def __getattr__(name):
    # Expose live DB state variables instead of stale import-time snapshots.
    if name in {"_GROOT", "_UROOT", "_VTRACKS"}:
        return getattr(_shared, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configuration
    'CONFIG',

    # Database functions
    'gdb_init',
    'gdb_reload',
    'gdb_unload',
    'gdb_info',
    'gdb_export_fasta',
    'gdb_examples_path',
    'gdb_init_examples',
    'gsetroot',
    'gdb_create',
    'gdb_create_genome',
    'gdb_create_linked',
    'gdb_convert_to_indexed',
    'gdb_get_readonly_attrs',
    'gdb_set_readonly_attrs',
    'gdataset_load',
    'gdataset_unload',
    'gdataset_ls',
    'gdataset_save',
    'gdataset_info',

    # Track functions
    'gextract',
    'gscreen',
    'gsummary',
    'gquantiles',
    'gdist',
    'gpartition',
    'gsample',
    'gcor',
    'gbins_summary',
    'gbins_quantiles',
    'gcis_decay',
    'gsegment',
    'gwilcox',
    'gtrack_ls',
    'gtrack_info',
    'gtrack_exists',
    'gtrack_dataset',
    'gtrack_create',
    'gtrack_create_dense',
    'gtrack_create_sparse',
    'gtrack_import',
    'gtrack_import_mappedseq',
    'gtrack_import_set',
    'gtrack_rm',
    'gtrack_mv',
    'gtrack_copy',
    'gtrack_convert_to_indexed',
    'gtrack_create_empty_indexed',
    'gtrack_attr_get',
    'gtrack_attr_set',
    'gtrack_attr_export',
    'gtrack_attr_import',
    'gtrack_var_ls',
    'gtrack_var_get',
    'gtrack_var_set',
    'gtrack_var_rm',
    'gtrack_modify',
    'gtrack_smooth',
    'gtrack_2d_create',
    'gtrack_2d_import',
    'gtrack_2d_import_contacts',
    'gtrack_create_pwm_energy',

    # Interval functions
    'gintervals',
    'gintervals_all',
    'gintervals_2d',
    'gintervals_2d_all',
    'gintervals_2d_band_intersect',
    'gintervals_union',
    'gintervals_intersect',
    'gintervals_diff',
    'gintervals_canonic',
    'gintervals_force_range',
    'gintervals_summary',
    'gintervals_quantiles',
    'gintervals_covered_bp',
    'gintervals_coverage_fraction',
    'gintervals_neighbors',
    'gintervals_neighbors_upstream',
    'gintervals_neighbors_downstream',
    'gintervals_neighbors_directional',
    'gintervals_from_tuples',
    'gintervals_from_strings',
    'gintervals_from_bed',
    'gintervals_import_genes',
    'gintervals_window',
    'gintervals_ls',
    'gintervals_exists',
    'gintervals_is_bigset',
    'gintervals_dataset',
    'gintervals_chrom_sizes',
    'gintervals_load',
    'gintervals_convert_to_indexed',
    'gintervals_2d_convert_to_indexed',
    'gintervals_is_indexed',
    'gintervals_save',
    'gintervals_update',
    'gintervals_mapply',
    'gintervals_rm',
    'giterator_cartesian_grid',
    'giterator_intervals',
    'gintervals_rbind',
    'gintervals_mark_overlaps',
    'gintervals_annotate',
    'gintervals_normalize',
    'gintervals_random',

    # Virtual track functions
    'gvtrack_create',
    'gvtrack_ls',
    'gvtrack_info',
    'gvtrack_iterator',
    'gvtrack_iterator_2d',
    'gvtrack_filter',
    'gvtrack_rm',
    'gvtrack_clear',

    # Sequence functions
    'gseq_extract',
    'gseq_rev',
    'gseq_comp',
    'gseq_revcomp',
    'gseq_kmer',
    'gseq_kmer_dist',
    'gseq_pwm',

    # Lookup functions
    'glookup',
    'gtrack_lookup',

    # Liftover functions
    'gintervals_load_chain',
    'gintervals_as_chain',
    'gintervals_liftover',
    'gtrack_liftover',

    # Directory management
    'gdir_cwd',
    'gdir_cd',
    'gdir_create',
    'gdir_rm',
    'gtrack_create_dirs',

    # Genome synthesis functions
    'GsynthModel',
    'gsynth_bin_map',
    'gsynth_train',
    'gsynth_sample',
    'gsynth_random',
    'gsynth_replace_kmer',
    'gsynth_save',
    'gsynth_load',

    # Internal (shared)
    '_bound_colname',
    '_checkroot',
    '_chunk_slices',
    '_df2pymisha',
    '_iterated_intervals',
    '_itr2pymisha',
    '_make_progress_callback',
    '_progress_context',
    '_pymisha',
    '_pymisha2df',
]

# Export module locals to the C extension for access to Python functions
# This must be at the end of the file after all functions are defined
_pymisha._PMLOCALS = locals()
