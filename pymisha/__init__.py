"""
PyMisha - Python wrapper for the misha Genomic Data Analysis Toolkit
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("pymisha")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Make numpy available for expressions
import numpy as np  # noqa: F401

from . import _shared
from ._iterator_policy import CartesianGridSpec, FixedRectPolicy, IntervalsPolicy, TrackRectsPolicy
from ._log import PymishaWarning
from ._shared import (  # noqa: F401 — imported for _PMLOCALS C++ bridge
    CONFIG,
    _checkroot,
    _chunk_slices,
    _df2pymisha,
    _iterated_intervals,
    _itr2pymisha,
    _make_progress_callback,
    _progress_context,
    _pymisha,
    _pymisha2df,
    gmax_processes,
)
from .analysis import gcis_decay, gcompute_strands_autocorr, gsegment, gwilcox
from .dataset import (
    gdataset_example_path,
    gdataset_info,
    gdataset_load,
    gdataset_ls,
    gdataset_save,
    gdataset_unload,
)
from .db import (
    gdb_examples_path,
    gdb_export_fasta,
    gdb_info,
    gdb_init,
    gdb_init_examples,
    gdb_mark_cache_dirty,
    gdb_reload,
    gdb_unload,
    gsetroot,
)
from .db_attrs import gdb_get_readonly_attrs, gdb_set_readonly_attrs
from .db_create import gdb_convert_to_indexed, gdb_create, gdb_create_genome, gdb_create_linked
from .extract import gextract, giterator_intervals_2d, gscreen
from .gdir import (
    gdir_cd,
    gdir_create,
    gdir_cwd,
    gdir_rm,
    gtrack_create_dirs,
)
from .genome import gdb_build_genome, gdb_install_intervals
from .genome.registry import gdb_genome_info, gdb_list_genomes
from .genome_edit import (
    ggenome_implant,
    ggenome_transplant,
)
from .gsynth import (
    GsynthModel,
    gsynth_bin_map,
    gsynth_cell_merge,
    gsynth_convert,
    gsynth_forbid_kmer,
    gsynth_load,
    gsynth_random,
    gsynth_replace_kmer,
    gsynth_sample,
    gsynth_save,
    gsynth_score,
    gsynth_train,
)
from .intervals import (
    gintervals,
    gintervals_2d,
    gintervals_2d_all,
    gintervals_2d_band_intersect,
    gintervals_2d_convert_to_indexed,
    gintervals_2d_intersect,
    gintervals_2d_union,
    gintervals_all,
    gintervals_annotate,
    gintervals_canonic,
    gintervals_chrom_sizes,
    gintervals_convert_to_indexed,
    gintervals_coverage_fraction,
    gintervals_covered_bp,
    gintervals_dataset,
    gintervals_dbs,
    gintervals_diff,
    gintervals_exists,
    gintervals_force_range,
    gintervals_from_bed,
    gintervals_from_strings,
    gintervals_from_tuples,
    gintervals_import_bed,
    gintervals_import_genes,
    gintervals_import_gff,
    gintervals_import_vcf,
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
    gintervals_path,
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
from .intervals_attr import (
    gintervals_attr_export,
    gintervals_attr_get,
    gintervals_attr_import,
    gintervals_attr_set,
)
from .intervals_mat import gintervals_from_mat, gintervals_to_mat
from .liftover import (
    gintervals_as_chain,
    gintervals_liftover,
    gintervals_load_chain,
    gtrack_liftover,
)
from .lookup import glookup, gtrack_lookup
from .motif_import import gseq_read_homer, gseq_read_jaspar, gseq_read_meme
from .sequence import (
    grevcomp,
    gseq_comp,
    gseq_extract,
    gseq_kmer,
    gseq_kmer_dist,
    gseq_pwm,
    gseq_pwm_edits,
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
from .track_export import gtrack_export_bedgraph, gtrack_export_bigwig
from .tracks import (
    gtrack_2d_convert_to_indexed,
    gtrack_2d_create,
    gtrack_2d_get_insu_borders,
    gtrack_2d_get_insu_doms,
    gtrack_2d_import,
    gtrack_2d_import_contacts,
    gtrack_array_create,
    gtrack_array_extract,
    gtrack_array_get_colnames,
    gtrack_array_import,
    gtrack_array_set_colnames,
    gtrack_attr_export,
    gtrack_attr_get,
    gtrack_attr_import,
    gtrack_attr_set,
    gtrack_convert_to_indexed,
    gtrack_copy,
    gtrack_create,
    gtrack_create_dense,
    gtrack_create_dense_direct,
    gtrack_create_empty_indexed,
    gtrack_create_pwm_energy,
    gtrack_create_sparse,
    gtrack_dataset,
    gtrack_dbs,
    gtrack_exists,
    gtrack_import,
    gtrack_import_mappedseq,
    gtrack_import_set,
    gtrack_info,
    gtrack_ls,
    gtrack_modify,
    gtrack_mv,
    gtrack_path,
    gtrack_rm,
    gtrack_smooth,
    gtrack_var_get,
    gtrack_var_ls,
    gtrack_var_rm,
    gtrack_var_set,
)
from .vtracks import (
    gvtrack_array_slice,
    gvtrack_clear,
    gvtrack_create,
    gvtrack_filter,
    gvtrack_info,
    gvtrack_iterator,
    gvtrack_iterator_2d,
    gvtrack_ls,
    gvtrack_rm,
)

# The C extension names this class "pymisha.error", so pickle (and hence
# multiprocessing error propagation) looks it up here. Without the alias any
# misha error raised inside a worker is replaced by an opaque PicklingError.
error = _pymisha.error

# Monkey-patch _pymisha.pm_dbreload so ANY caller (including tests that
# import the C extension directly) triggers Python-side cache
# invalidation.  The C++ side never mutates the track scan without
# pm_dbreload (or pm_dbinit / pm_dbunload) being called, so wrapping
# this entry point keeps the Python caches (track_names, computed-track
# types, expr-validation set) consistent with the C++ track_cache.
_original_pm_dbreload = _pymisha.pm_dbreload


def _pm_dbreload_with_invalidation(*args: Any, **kwargs: Any) -> Any:
    result = _original_pm_dbreload(*args, **kwargs)
    _shared._clear_track_names_cache()
    from .tracks import _clear_computed_track_cache
    _clear_computed_track_cache()
    # Track *content* caches. A track rewritten under the same name (or a
    # gtrack_rm + gtrack_create round trip) otherwise keeps serving values
    # derived from the old data: global.percentile silently returns ranks
    # against a stale reference, and the 2D reader keeps an mmap of the
    # deleted track.dat.
    from ._quadtree import clear_indexed_2d_cache
    from .vtracks import _GLOBAL_PERCENTILE_CACHE, _PV_TABLE_CACHE
    _GLOBAL_PERCENTILE_CACHE.clear()
    _PV_TABLE_CACHE.clear()
    clear_indexed_2d_cache()
    return result


_pymisha.pm_dbreload = _pm_dbreload_with_invalidation


def __getattr__(name: str) -> Any:
    # Expose live DB state variables instead of stale import-time snapshots.
    if name in {"_GROOT", "_UROOT", "_VTRACKS"}:
        return getattr(_shared, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Configuration
    'CONFIG',
    'gmax_processes',
    'PymishaWarning',
    'error',

    # Database functions
    'gdb_init',
    'gdb_reload',
    'gdb_unload',
    'gdb_info',
    'gdb_mark_cache_dirty',
    'gdb_export_fasta',
    'gdb_examples_path',
    'gdb_init_examples',
    'gsetroot',
    'gdb_create',
    'gdb_create_genome',
    'gdb_create_linked',
    'gdb_build_genome',
    'gdb_genome_info',
    'gdb_install_intervals',
    'gdb_list_genomes',
    'gdb_convert_to_indexed',
    'gdb_get_readonly_attrs',
    'gdb_set_readonly_attrs',
    'gdataset_example_path',
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
    'gcompute_strands_autocorr',
    'gsegment',
    'gwilcox',
    'gtrack_dbs',
    'gtrack_ls',
    'gtrack_info',
    'gtrack_exists',
    'gtrack_path',
    'gtrack_dataset',
    'gtrack_create',
    'gtrack_create_dense',
    'gtrack_create_dense_direct',
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
    'gtrack_2d_convert_to_indexed',
    'gtrack_2d_create',
    'gtrack_2d_get_insu_borders',
    'gtrack_2d_get_insu_doms',
    'gtrack_2d_import',
    'gtrack_2d_import_contacts',
    'gtrack_array_create',
    'gtrack_array_extract',
    'gtrack_array_get_colnames',
    'gtrack_array_import',
    'gtrack_array_set_colnames',
    'gtrack_create_pwm_energy',
    'gtrack_export_bedgraph',
    'gtrack_export_bigwig',

    # Interval functions
    'gintervals',
    'gintervals_all',
    'gintervals_2d',
    'gintervals_2d_all',
    'gintervals_2d_band_intersect',
    'gintervals_2d_intersect',
    'gintervals_2d_union',
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
    'gintervals_import_bed',
    'gintervals_import_genes',
    'gintervals_import_gff',
    'gintervals_import_vcf',
    'gintervals_window',
    'gintervals_dbs',
    'gintervals_ls',
    'gintervals_exists',
    'gintervals_path',
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
    'gintervals_attr_get',
    'gintervals_attr_set',
    'gintervals_attr_export',
    'gintervals_attr_import',
    'giterator_cartesian_grid',
    'giterator_intervals',
    'giterator_intervals_2d',
    'CartesianGridSpec',
    'FixedRectPolicy',
    'IntervalsPolicy',
    'TrackRectsPolicy',
    'gintervals_rbind',
    'gintervals_mark_overlaps',
    'gintervals_annotate',
    'gintervals_normalize',
    'gintervals_random',
    'gintervals_to_mat',
    'gintervals_from_mat',

    # Virtual track functions
    'gvtrack_array_slice',
    'gvtrack_create',
    'gvtrack_ls',
    'gvtrack_info',
    'gvtrack_iterator',
    'gvtrack_iterator_2d',
    'gvtrack_filter',
    'gvtrack_rm',
    'gvtrack_clear',

    # Sequence functions
    'grevcomp',
    'gseq_extract',
    'gseq_rev',
    'gseq_comp',
    'gseq_revcomp',
    'gseq_kmer',
    'gseq_kmer_dist',
    'gseq_pwm',
    'gseq_pwm_edits',
    'gseq_read_meme',
    'gseq_read_jaspar',
    'gseq_read_homer',

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
    'gsynth_cell_merge',
    'gsynth_train',
    'gsynth_sample',
    'gsynth_score',
    'gsynth_random',
    'gsynth_replace_kmer',
    'gsynth_forbid_kmer',
    'gsynth_save',
    'gsynth_load',
    'gsynth_convert',

    # Genome editing functions
    'ggenome_implant',
    'ggenome_transplant',

]

# Bridge the Python module namespace to the C++ extension. The C++ side
# looks up Python functions by name at runtime (e.g., for interval callbacks
# and expression evaluation). locals() captures every public function, class,
# and import defined above. Monkeypatching after import will be visible to C++.
# This line MUST remain at the end of the file — moving it earlier means the
# C++ extension sees an incomplete namespace.
_pymisha._PMLOCALS = locals()
