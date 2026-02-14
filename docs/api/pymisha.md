# API Reference

## Database

Functions for initializing, configuring, and managing genomic databases, including directory operations and genome creation.

::: pymisha.gdb_init
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsetroot
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_reload
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_unload
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_info
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_examples_path
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_init_examples
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_create
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_create_genome
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_create_linked
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_convert_to_indexed
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_get_readonly_attrs
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdb_set_readonly_attrs
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdir_cwd
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdir_cd
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdir_create
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdir_rm
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_create_dirs
    options:
      show_root_heading: true
      heading_level: 3

---

## Datasets

Functions for loading, saving, and managing named datasets within the genomic database.

::: pymisha.gdataset_load
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdataset_unload
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdataset_ls
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdataset_save
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdataset_info
    options:
      show_root_heading: true
      heading_level: 3

---

## Tracks

Functions for creating, importing, modifying, and managing genomic tracks, including dense, sparse, 2D, and indexed track types, as well as track attributes and variables.

::: pymisha.gtrack_ls
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_info
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_exists
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_dataset
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_create
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_create_dense
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_create_sparse
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_create_empty_indexed
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_import
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_import_mappedseq
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_import_set
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_rm
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_mv
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_copy
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_modify
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_smooth
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_convert_to_indexed
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_2d_create
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_2d_import
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_2d_import_contacts
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_create_pwm_energy
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_attr_get
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_attr_set
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_attr_export
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_attr_import
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_var_ls
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_var_get
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_var_set
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_var_rm
    options:
      show_root_heading: true
      heading_level: 3

---

## Virtual Tracks

Functions for creating and managing virtual tracks, which define computed views over existing tracks with custom iterators and filters.

::: pymisha.gvtrack_create
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gvtrack_ls
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gvtrack_info
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gvtrack_iterator
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gvtrack_iterator_2d
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gvtrack_filter
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gvtrack_rm
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gvtrack_clear
    options:
      show_root_heading: true
      heading_level: 3

---

## Intervals

Functions for creating, manipulating, and querying genomic intervals, including set operations, annotation, normalization, and I/O.

::: pymisha.gintervals
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_all
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_2d
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_2d_all
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_2d_band_intersect
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_union
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_intersect
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_diff
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_canonic
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_force_range
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_covered_bp
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_coverage_fraction
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_mark_overlaps
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_annotate
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_normalize
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_neighbors
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_neighbors_upstream
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_neighbors_downstream
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_neighbors_directional
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_random
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_from_tuples
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_from_strings
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_from_bed
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_import_genes
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_window
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_ls
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_exists
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_dataset
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_chrom_sizes
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_load
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_save
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_update
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_rm
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_rbind
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_mapply
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_convert_to_indexed
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_2d_convert_to_indexed
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_is_indexed
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.giterator_cartesian_grid
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.giterator_intervals
    options:
      show_root_heading: true
      heading_level: 3

---

## Data Operations

Functions for extracting, summarizing, and analyzing track data across genomic intervals, including statistical summaries, distributions, correlations, and segmentation.

::: pymisha.gextract
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gscreen
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsummary
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gquantiles
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gdist
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gpartition
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsample
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gcor
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gbins_summary
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gbins_quantiles
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_summary
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_quantiles
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gcis_decay
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsegment
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gwilcox
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.glookup
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_lookup
    options:
      show_root_heading: true
      heading_level: 3

---

## Liftover

Functions for converting genomic coordinates between assemblies using chain files.

::: pymisha.gintervals_load_chain
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_as_chain
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gintervals_liftover
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_liftover
    options:
      show_root_heading: true
      heading_level: 3

---

## Sequence Analysis

Functions for extracting and analyzing DNA sequences, including reverse complement operations, k-mer counting, and position weight matrix scoring.

::: pymisha.gseq_extract
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gseq_rev
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gseq_comp
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gseq_revcomp
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gseq_kmer
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gseq_kmer_dist
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gseq_pwm
    options:
      show_root_heading: true
      heading_level: 3

---

## Genome Synthesis

Classes and functions for training generative models on genomic sequences and sampling synthetic genomes.

::: pymisha.GsynthModel
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsynth_bin_map
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsynth_train
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsynth_sample
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsynth_random
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsynth_replace_kmer
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsynth_save
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gsynth_load
    options:
      show_root_heading: true
      heading_level: 3
