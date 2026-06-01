# Tracks

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

::: pymisha.gtrack_2d_convert_to_indexed
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

## Array Tracks

Multi-column per-position tracks. The PyMisha implementation reads and writes
the on-disk format byte-compatibly with R misha. Array tracks dispatch through
the C++ track-expression scanner, so they work directly in expressions
(`gextract("my.array", ...)`, `gextract("2 * my.array", ...)`), as an iterator,
and as the source of value / slice virtual tracks. Use `gtrack_array_extract`
to dump the full multi-column matrix.

::: pymisha.gtrack_array_create
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_array_extract
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_array_get_colnames
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_array_set_colnames
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

## Track Export

Functions to export tracks to standard genomic file formats.

::: pymisha.gtrack_export_bedgraph
    options:
      show_root_heading: true
      heading_level: 3

::: pymisha.gtrack_export_bigwig
    options:
      show_root_heading: true
      heading_level: 3
