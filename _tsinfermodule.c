
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <float.h>
#include <stdbool.h>

#include "lib/tsinfer.h"

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#define MODULE_DOC \
"Low-level tsinfer interface."

static PyObject *TsinfLibraryError;

/* A lightweight wrapper for a table collection. This serves only as a wrapper
 * around a pointer and a way move to data in-and-out of the low level structures
 * via the canonical dictionary encoding.
 *
 * Copied from _msprimemodule 2020-03-02
 * Originally copied from _tskitmodule.c 2018-12-20.
 */
typedef struct {
    PyObject_HEAD
    tsk_table_collection_t *tables;
} LightweightTableCollection;

typedef struct {
    PyObject_HEAD
    ancestor_builder_t *builder;
} AncestorBuilder;

typedef struct {
    PyObject_HEAD
    tree_sequence_builder_t *tree_sequence_builder;
} TreeSequenceBuilder;

typedef struct {
    PyObject_HEAD
    ancestor_matcher_t *ancestor_matcher;
    TreeSequenceBuilder *tree_sequence_builder;
} AncestorMatcher;

static void
handle_library_error(int err)
{
    if (err == TSI_ERR_NO_MEMORY) {
        PyErr_NoMemory();
    } else {
        PyErr_Format(TsinfLibraryError, "Error occured: %d", err);
    }
}

/*===================================================================
 * General table code.
 *===================================================================
 */

/* NOTE: this code was copied from _tskitmodule as the efficient way to
 * import and export TableCollection data. It is unlikely to change
 * much over time, but if updates need to be made it would be better
 * to copy the code wholesale. The tests in ``test_dict_encoding.py``
 * are designed to test this code thoroughly, and also come from
 * tskit.
 */

/*
 * Retrieves the PyObject* corresponding the specified key in the
 * specified dictionary. If required is true, raise a TypeError if the
 * value is None.
 *
 * NB This returns a *borrowed reference*, so don't DECREF it!
 */
static PyObject *
get_table_dict_value(PyObject *dict, const char *key_str, bool required)
{
    PyObject *ret = NULL;

    ret = PyDict_GetItemString(dict, key_str);
    if (ret == NULL) {
        PyErr_Format(PyExc_ValueError, "'%s' not specified", key_str);
    }
    if (required && ret == Py_None) {
        PyErr_Format(PyExc_TypeError, "'%s' is required", key_str);
        ret = NULL;
    }
    return ret;
}

static PyArrayObject *
table_read_column_array(PyObject *input, int npy_type, size_t *num_rows, bool check_num_rows)
{
    PyArrayObject *ret = NULL;
    PyArrayObject *array = NULL;
    npy_intp *shape;

    array = (PyArrayObject *) PyArray_FROMANY(input, npy_type, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(array);
    if (check_num_rows) {
        if (*num_rows != (size_t) shape[0]) {
            PyErr_SetString(PyExc_ValueError, "Input array dimensions must be equal.");
            goto out;
        }
    } else {
        *num_rows = (size_t) shape[0];
    }
    ret = array;
    array = NULL;
out:
    Py_XDECREF(array);
    return ret;
}

static PyArrayObject *
table_read_offset_array(PyObject *input, size_t *num_rows, size_t length, bool check_num_rows)
{
    PyArrayObject *ret = NULL;
    PyArrayObject *array = NULL;
    npy_intp *shape;
    uint32_t *data;

    array = (PyArrayObject *) PyArray_FROMANY(input, NPY_UINT32, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(array);
    if (! check_num_rows) {
        *num_rows = (size_t) shape[0];
        if (*num_rows == 0) {
            PyErr_SetString(PyExc_ValueError, "Offset arrays must have at least one element");
            goto out;
        }
        *num_rows -= 1;
    }
    if (((size_t) shape[0]) != *num_rows + 1) {
        PyErr_SetString(PyExc_ValueError, "offset columns must have n + 1 rows.");
        goto out;
    }
    data = PyArray_DATA(array);
    if (data[*num_rows] != (uint32_t) length) {
        PyErr_SetString(PyExc_ValueError, "Bad offset column encoding");
        goto out;
    }
    ret = array;
out:
    if (ret == NULL) {
        Py_XDECREF(array);
    }
    return ret;
}

static int
parse_individual_table_dict(tsk_individual_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, metadata_length, location_length;
    char *metadata_data = NULL;
    double *location_data = NULL;
    uint32_t *metadata_offset_data = NULL;
    uint32_t *location_offset_data = NULL;
    PyObject *flags_input = NULL;
    PyArrayObject *flags_array = NULL;
    PyObject *location_input = NULL;
    PyArrayObject *location_array = NULL;
    PyObject *location_offset_input = NULL;
    PyArrayObject *location_offset_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;

    /* Get the input values */
    flags_input = get_table_dict_value(dict, "flags", true);
    if (flags_input == NULL) {
        goto out;
    }
    location_input = get_table_dict_value(dict, "location", false);
    if (location_input == NULL) {
        goto out;
    }
    location_offset_input = get_table_dict_value(dict, "location_offset", false);
    if (location_offset_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }

    /* Pull out the arrays */
    flags_array = table_read_column_array(flags_input, NPY_UINT32, &num_rows, false);
    if (flags_array == NULL) {
        goto out;
    }
    if ((location_input == Py_None) != (location_offset_input == Py_None)) {
        PyErr_SetString(PyExc_TypeError,
                "location and location_offset must be specified together");
        goto out;
    }
    if (location_input != Py_None) {
        location_array = table_read_column_array(location_input, NPY_FLOAT64,
                &location_length, false);
        if (location_array == NULL) {
            goto out;
        }
        location_data = PyArray_DATA(location_array);
        location_offset_array = table_read_offset_array(location_offset_input, &num_rows,
                location_length, true);
        if (location_offset_array == NULL) {
            goto out;
        }
        location_offset_data = PyArray_DATA(location_offset_array);
    }
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(PyExc_TypeError,
                "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array = table_read_column_array(metadata_input, NPY_INT8,
                &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(metadata_offset_input, &num_rows,
                metadata_length, true);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }

    if (clear_table) {
        err = tsk_individual_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_individual_table_append_columns(table, num_rows,
            PyArray_DATA(flags_array),
            location_data, location_offset_data,
            metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(flags_array);
    Py_XDECREF(location_array);
    Py_XDECREF(location_offset_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_node_table_dict(tsk_node_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, metadata_length;
    char *metadata_data = NULL;
    uint32_t *metadata_offset_data = NULL;
    void *population_data = NULL;
    void *individual_data = NULL;
    PyObject *time_input = NULL;
    PyArrayObject *time_array = NULL;
    PyObject *flags_input = NULL;
    PyArrayObject *flags_array = NULL;
    PyObject *population_input = NULL;
    PyArrayObject *population_array = NULL;
    PyObject *individual_input = NULL;
    PyArrayObject *individual_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;

    /* Get the input values */
    flags_input = get_table_dict_value(dict, "flags", true);
    if (flags_input == NULL) {
        goto out;
    }
    time_input = get_table_dict_value(dict, "time", true);
    if (time_input == NULL) {
        goto out;
    }
    population_input = get_table_dict_value(dict, "population", false);
    if (population_input == NULL) {
        goto out;
    }
    individual_input = get_table_dict_value(dict, "individual", false);
    if (individual_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }

    /* Create the arrays */
    flags_array = table_read_column_array(flags_input, NPY_UINT32, &num_rows, false);
    if (flags_array == NULL) {
        goto out;
    }
    time_array = table_read_column_array(time_input, NPY_FLOAT64, &num_rows, true);
    if (time_array == NULL) {
        goto out;
    }
    if (population_input != Py_None) {
        population_array = table_read_column_array(population_input, NPY_INT32,
                &num_rows, true);
        if (population_array == NULL) {
            goto out;
        }
        population_data = PyArray_DATA(population_array);
    }
    if (individual_input != Py_None) {
        individual_array = table_read_column_array(individual_input, NPY_INT32,
                &num_rows, true);
        if (individual_array == NULL) {
            goto out;
        }
        individual_data = PyArray_DATA(individual_array);
    }
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(PyExc_TypeError,
                "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array = table_read_column_array(metadata_input, NPY_INT8,
                &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(metadata_offset_input, &num_rows,
                metadata_length, true);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }
    if (clear_table) {
        err = tsk_node_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_node_table_append_columns(table, num_rows,
            PyArray_DATA(flags_array), PyArray_DATA(time_array), population_data,
            individual_data, metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(flags_array);
    Py_XDECREF(time_array);
    Py_XDECREF(population_array);
    Py_XDECREF(individual_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_edge_table_dict(tsk_edge_table_t *table, PyObject *dict, bool clear_table)
{
    int ret = -1;
    int err;
    size_t num_rows = 0;
    PyObject *left_input = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right_input = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent_input = NULL;
    PyArrayObject *parent_array = NULL;
    PyObject *child_input = NULL;
    PyArrayObject *child_array = NULL;

    /* Get the input values */
    left_input = get_table_dict_value(dict, "left", true);
    if (left_input == NULL) {
        goto out;
    }
    right_input = get_table_dict_value(dict, "right", true);
    if (right_input == NULL) {
        goto out;
    }
    parent_input = get_table_dict_value(dict, "parent", true);
    if (parent_input == NULL) {
        goto out;
    }
    child_input = get_table_dict_value(dict, "child", true);
    if (child_input == NULL) {
        goto out;
    }

    /* Create the arrays */
    left_array = table_read_column_array(left_input, NPY_FLOAT64, &num_rows, false);
    if (left_array == NULL) {
        goto out;
    }
    right_array = table_read_column_array(right_input, NPY_FLOAT64, &num_rows, true);
    if (right_array == NULL) {
        goto out;
    }
    parent_array = table_read_column_array(parent_input, NPY_INT32, &num_rows, true);
    if (parent_array == NULL) {
        goto out;
    }
    child_array = table_read_column_array(child_input, NPY_INT32, &num_rows, true);
    if (child_array == NULL) {
        goto out;
    }

    if (clear_table) {
        err = tsk_edge_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_edge_table_append_columns(table, num_rows,
            PyArray_DATA(left_array), PyArray_DATA(right_array),
            PyArray_DATA(parent_array), PyArray_DATA(child_array));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(parent_array);
    Py_XDECREF(child_array);
    return ret;
}

static int
parse_migration_table_dict(tsk_migration_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows;
    PyObject *left_input = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right_input = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *node_input = NULL;
    PyArrayObject *node_array = NULL;
    PyObject *source_input = NULL;
    PyArrayObject *source_array = NULL;
    PyObject *dest_input = NULL;
    PyArrayObject *dest_array = NULL;
    PyObject *time_input = NULL;
    PyArrayObject *time_array = NULL;

    /* Get the input values */
    left_input = get_table_dict_value(dict, "left", true);
    if (left_input == NULL) {
        goto out;
    }
    right_input = get_table_dict_value(dict, "right", true);
    if (right_input == NULL) {
        goto out;
    }
    node_input = get_table_dict_value(dict, "node", true);
    if (node_input == NULL) {
        goto out;
    }
    source_input = get_table_dict_value(dict, "source", true);
    if (source_input == NULL) {
        goto out;
    }
    dest_input = get_table_dict_value(dict, "dest", true);
    if (dest_input == NULL) {
        goto out;
    }
    time_input = get_table_dict_value(dict, "time", true);
    if (time_input == NULL) {
        goto out;
    }

    /* Build the arrays */
    left_array = table_read_column_array(left_input, NPY_FLOAT64, &num_rows, false);
    if (left_array == NULL) {
        goto out;
    }
    right_array = table_read_column_array(right_input, NPY_FLOAT64, &num_rows, true);
    if (right_array == NULL) {
        goto out;
    }
    node_array = table_read_column_array(node_input, NPY_INT32, &num_rows, true);
    if (node_array == NULL) {
        goto out;
    }
    source_array = table_read_column_array(source_input, NPY_INT32, &num_rows, true);
    if (source_array == NULL) {
        goto out;
    }
    dest_array = table_read_column_array(dest_input, NPY_INT32, &num_rows, true);
    if (dest_array == NULL) {
        goto out;
    }
    time_array = table_read_column_array(time_input, NPY_FLOAT64, &num_rows, true);
    if (time_array == NULL) {
        goto out;
    }

    if (clear_table) {
        err = tsk_migration_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_migration_table_append_columns(table, num_rows,
        PyArray_DATA(left_array), PyArray_DATA(right_array), PyArray_DATA(node_array),
        PyArray_DATA(source_array), PyArray_DATA(dest_array), PyArray_DATA(time_array));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(node_array);
    Py_XDECREF(source_array);
    Py_XDECREF(dest_array);
    Py_XDECREF(time_array);
    return ret;
}

static int
parse_site_table_dict(tsk_site_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows = 0;
    size_t ancestral_state_length, metadata_length;
    PyObject *position_input = NULL;
    PyArrayObject *position_array = NULL;
    PyObject *ancestral_state_input = NULL;
    PyArrayObject *ancestral_state_array = NULL;
    PyObject *ancestral_state_offset_input = NULL;
    PyArrayObject *ancestral_state_offset_array = NULL;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    char *metadata_data;
    uint32_t *metadata_offset_data;

    /* Get the input values */
    position_input = get_table_dict_value(dict, "position", true);
    if (position_input == NULL) {
        goto out;
    }
    ancestral_state_input = get_table_dict_value(dict, "ancestral_state", true);
    if (ancestral_state_input == NULL) {
        goto out;
    }
    ancestral_state_offset_input = get_table_dict_value(dict, "ancestral_state_offset", true);
    if (ancestral_state_offset_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }

    /* Get the arrays */
    position_array = table_read_column_array(position_input, NPY_FLOAT64, &num_rows, false);
    if (position_array == NULL) {
        goto out;
    }
    ancestral_state_array = table_read_column_array(ancestral_state_input, NPY_INT8,
            &ancestral_state_length, false);
    if (ancestral_state_array == NULL) {
        goto out;
    }
    ancestral_state_offset_array = table_read_offset_array(ancestral_state_offset_input,
            &num_rows, ancestral_state_length, true);
    if (ancestral_state_offset_array == NULL) {
        goto out;
    }

    metadata_data = NULL;
    metadata_offset_data = NULL;
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(PyExc_TypeError,
                "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array = table_read_column_array(metadata_input, NPY_INT8,
                &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(metadata_offset_input, &num_rows,
                metadata_length, false);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }

    if (clear_table) {
        err = tsk_site_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_site_table_append_columns(table, num_rows,
        PyArray_DATA(position_array), PyArray_DATA(ancestral_state_array),
        PyArray_DATA(ancestral_state_offset_array), metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(position_array);
    Py_XDECREF(ancestral_state_array);
    Py_XDECREF(ancestral_state_offset_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_mutation_table_dict(tsk_mutation_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows = 0;
    size_t derived_state_length = 0;
    size_t metadata_length = 0;
    PyObject *site_input = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *derived_state_input = NULL;
    PyArrayObject *derived_state_array = NULL;
    PyObject *derived_state_offset_input = NULL;
    PyArrayObject *derived_state_offset_array = NULL;
    PyObject *node_input = NULL;
    PyArrayObject *node_array = NULL;
    PyObject *parent_input = NULL;
    PyArrayObject *parent_array = NULL;
    tsk_id_t *parent_data;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;
    char *metadata_data;
    uint32_t *metadata_offset_data;

    /* Get the input values */
    site_input = get_table_dict_value(dict, "site", true);
    if (site_input == NULL) {
        goto out;
    }
    node_input = get_table_dict_value(dict, "node", true);
    if (node_input == NULL) {
        goto out;
    }
    parent_input = get_table_dict_value(dict, "parent", false);
    if (parent_input == NULL) {
        goto out;
    }
    derived_state_input = get_table_dict_value(dict, "derived_state", true);
    if (derived_state_input == NULL) {
        goto out;
    }
    derived_state_offset_input = get_table_dict_value(dict, "derived_state_offset", true);
    if (derived_state_offset_input == NULL) {
        goto out;
    }
    metadata_input = get_table_dict_value(dict, "metadata", false);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", false);
    if (metadata_offset_input == NULL) {
        goto out;
    }

    /* Get the arrays */
    site_array = table_read_column_array(site_input, NPY_INT32, &num_rows, false);
    if (site_array == NULL) {
        goto out;
    }
    derived_state_array = table_read_column_array(derived_state_input, NPY_INT8,
            &derived_state_length, false);
    if (derived_state_array == NULL) {
        goto out;
    }
    derived_state_offset_array = table_read_offset_array(derived_state_offset_input,
            &num_rows, derived_state_length, true);
    if (derived_state_offset_array == NULL) {
        goto out;
    }
    node_array = table_read_column_array(node_input, NPY_INT32, &num_rows, true);
    if (node_array == NULL) {
        goto out;
    }

    parent_data = NULL;
    if (parent_input != Py_None) {
        parent_array = table_read_column_array(parent_input, NPY_INT32, &num_rows, true);
        if (parent_array == NULL) {
            goto out;
        }
        parent_data = PyArray_DATA(parent_array);
    }

    metadata_data = NULL;
    metadata_offset_data = NULL;
    if ((metadata_input == Py_None) != (metadata_offset_input == Py_None)) {
        PyErr_SetString(PyExc_TypeError,
                "metadata and metadata_offset must be specified together");
        goto out;
    }
    if (metadata_input != Py_None) {
        metadata_array = table_read_column_array(metadata_input, NPY_INT8,
                &metadata_length, false);
        if (metadata_array == NULL) {
            goto out;
        }
        metadata_data = PyArray_DATA(metadata_array);
        metadata_offset_array = table_read_offset_array(metadata_offset_input, &num_rows,
                metadata_length, false);
        if (metadata_offset_array == NULL) {
            goto out;
        }
        metadata_offset_data = PyArray_DATA(metadata_offset_array);
    }

    if (clear_table) {
        err = tsk_mutation_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_mutation_table_append_columns(table, num_rows,
            PyArray_DATA(site_array), PyArray_DATA(node_array),
            parent_data, PyArray_DATA(derived_state_array),
            PyArray_DATA(derived_state_offset_array),
            metadata_data, metadata_offset_data);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(site_array);
    Py_XDECREF(derived_state_array);
    Py_XDECREF(derived_state_offset_array);
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    Py_XDECREF(node_array);
    Py_XDECREF(parent_array);
    return ret;
}

static int
parse_population_table_dict(tsk_population_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, metadata_length;
    PyObject *metadata_input = NULL;
    PyArrayObject *metadata_array = NULL;
    PyObject *metadata_offset_input = NULL;
    PyArrayObject *metadata_offset_array = NULL;

    /* Get the inputs */
    metadata_input = get_table_dict_value(dict, "metadata", true);
    if (metadata_input == NULL) {
        goto out;
    }
    metadata_offset_input = get_table_dict_value(dict, "metadata_offset", true);
    if (metadata_offset_input == NULL) {
        goto out;
    }

    /* Get the arrays */
    metadata_array = table_read_column_array(metadata_input, NPY_INT8,
            &metadata_length, false);
    if (metadata_array == NULL) {
        goto out;
    }
    metadata_offset_array = table_read_offset_array(metadata_offset_input, &num_rows,
            metadata_length, false);
    if (metadata_offset_array == NULL) {
        goto out;
    }

    if (clear_table) {
        err = tsk_population_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_population_table_append_columns(table, num_rows,
            PyArray_DATA(metadata_array), PyArray_DATA(metadata_offset_array));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(metadata_array);
    Py_XDECREF(metadata_offset_array);
    return ret;
}

static int
parse_provenance_table_dict(tsk_provenance_table_t *table, PyObject *dict, bool clear_table)
{
    int err;
    int ret = -1;
    size_t num_rows, timestamp_length, record_length;
    PyObject *timestamp_input = NULL;
    PyArrayObject *timestamp_array = NULL;
    PyObject *timestamp_offset_input = NULL;
    PyArrayObject *timestamp_offset_array = NULL;
    PyObject *record_input = NULL;
    PyArrayObject *record_array = NULL;
    PyObject *record_offset_input = NULL;
    PyArrayObject *record_offset_array = NULL;

    /* Get the inputs */
    timestamp_input = get_table_dict_value(dict, "timestamp", true);
    if (timestamp_input == NULL) {
        goto out;
    }
    timestamp_offset_input = get_table_dict_value(dict, "timestamp_offset", true);
    if (timestamp_offset_input == NULL) {
        goto out;
    }
    record_input = get_table_dict_value(dict, "record", true);
    if (record_input == NULL) {
        goto out;
    }
    record_offset_input = get_table_dict_value(dict, "record_offset", true);
    if (record_offset_input == NULL) {
        goto out;
    }

    timestamp_array = table_read_column_array(timestamp_input, NPY_INT8,
            &timestamp_length, false);
    if (timestamp_array == NULL) {
        goto out;
    }
    timestamp_offset_array = table_read_offset_array(timestamp_offset_input, &num_rows,
            timestamp_length, false);
    if (timestamp_offset_array == NULL) {
        goto out;
    }
    record_array = table_read_column_array(record_input, NPY_INT8,
            &record_length, false);
    if (record_array == NULL) {
        goto out;
    }
    record_offset_array = table_read_offset_array(record_offset_input, &num_rows,
            record_length, true);
    if (record_offset_array == NULL) {
        goto out;
    }

    if (clear_table) {
        err = tsk_provenance_table_clear(table);
        if (err != 0) {
            handle_library_error(err);
            goto out;
        }
    }
    err = tsk_provenance_table_append_columns(table, num_rows,
            PyArray_DATA(timestamp_array), PyArray_DATA(timestamp_offset_array),
            PyArray_DATA(record_array), PyArray_DATA(record_offset_array));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(timestamp_array);
    Py_XDECREF(timestamp_offset_array);
    Py_XDECREF(record_array);
    Py_XDECREF(record_offset_array);
    return ret;
}

static int
parse_table_collection_dict(tsk_table_collection_t *tables, PyObject *tables_dict)
{
    int ret = -1;
    PyObject *value = NULL;

    value = get_table_dict_value(tables_dict, "sequence_length", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyNumber_Check(value)) {
        PyErr_Format(PyExc_TypeError, "'sequence_length' is not number");
        goto out;
    }
    tables->sequence_length = PyFloat_AsDouble(value);

    /* individuals */
    value = get_table_dict_value(tables_dict, "individuals", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_individual_table_dict(&tables->individuals, value, true) != 0) {
        goto out;
    }

    /* nodes */
    value = get_table_dict_value(tables_dict, "nodes", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_node_table_dict(&tables->nodes, value, true) != 0) {
        goto out;
    }

    /* edges */
    value = get_table_dict_value(tables_dict, "edges", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_edge_table_dict(&tables->edges, value, true) != 0) {
        goto out;
    }

    /* migrations */
    value = get_table_dict_value(tables_dict, "migrations", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_migration_table_dict(&tables->migrations, value, true) != 0) {
        goto out;
    }

    /* sites */
    value = get_table_dict_value(tables_dict, "sites", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_site_table_dict(&tables->sites, value, true) != 0) {
        goto out;
    }

    /* mutations */
    value = get_table_dict_value(tables_dict, "mutations", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_mutation_table_dict(&tables->mutations, value, true) != 0) {
        goto out;
    }

    /* populations */
    value = get_table_dict_value(tables_dict, "populations", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_population_table_dict(&tables->populations, value, true) != 0) {
        goto out;
    }

    /* provenances */
    value = get_table_dict_value(tables_dict, "provenances", true);
    if (value == NULL) {
        goto out;
    }
    if (!PyDict_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "not a dictionary");
        goto out;
    }
    if (parse_provenance_table_dict(&tables->provenances, value, true) != 0) {
        goto out;
    }

    ret = 0;
out:
    return ret;
}

static int
write_table_arrays(tsk_table_collection_t *tables, PyObject *dict)
{
    struct table_col {
        const char *name;
        void *data;
        npy_intp num_rows;
        int type;
    };
    struct table_desc {
        const char *name;
        struct table_col *cols;
    };
    int ret = -1;
    PyObject *array = NULL;
    PyObject *table_dict = NULL;
    size_t j;
    struct table_col *col;

    struct table_col individual_cols[] = {
        {"flags",
            (void *) tables->individuals.flags, tables->individuals.num_rows, NPY_UINT32},
        {"location",
            (void *) tables->individuals.location, tables->individuals.location_length,
            NPY_FLOAT64},
        {"location_offset",
            (void *) tables->individuals.location_offset, tables->individuals.num_rows + 1,
            NPY_UINT32},
        {"metadata",
            (void *) tables->individuals.metadata, tables->individuals.metadata_length,
            NPY_INT8},
        {"metadata_offset",
            (void *) tables->individuals.metadata_offset, tables->individuals.num_rows + 1,
            NPY_UINT32},
        {NULL},
    };

    struct table_col node_cols[] = {
        {"time",
            (void *) tables->nodes.time, tables->nodes.num_rows, NPY_FLOAT64},
        {"flags",
            (void *) tables->nodes.flags, tables->nodes.num_rows, NPY_UINT32},
        {"population",
            (void *) tables->nodes.population, tables->nodes.num_rows, NPY_INT32},
        {"individual",
            (void *) tables->nodes.individual, tables->nodes.num_rows, NPY_INT32},
        {"metadata",
            (void *) tables->nodes.metadata, tables->nodes.metadata_length, NPY_INT8},
        {"metadata_offset",
            (void *) tables->nodes.metadata_offset, tables->nodes.num_rows + 1, NPY_UINT32},
        {NULL},
    };

    struct table_col edge_cols[] = {
        {"left", (void *) tables->edges.left, tables->edges.num_rows, NPY_FLOAT64},
        {"right", (void *) tables->edges.right, tables->edges.num_rows, NPY_FLOAT64},
        {"parent", (void *) tables->edges.parent, tables->edges.num_rows, NPY_INT32},
        {"child", (void *) tables->edges.child, tables->edges.num_rows, NPY_INT32},
        {NULL},
    };

    struct table_col migration_cols[] = {
        {"left",
            (void *) tables->migrations.left, tables->migrations.num_rows,  NPY_FLOAT64},
        {"right",
            (void *) tables->migrations.right, tables->migrations.num_rows,  NPY_FLOAT64},
        {"node",
            (void *) tables->migrations.node, tables->migrations.num_rows,  NPY_INT32},
        {"source",
            (void *) tables->migrations.source, tables->migrations.num_rows,  NPY_INT32},
        {"dest",
            (void *) tables->migrations.dest, tables->migrations.num_rows,  NPY_INT32},
        {"time",
            (void *) tables->migrations.time, tables->migrations.num_rows,  NPY_FLOAT64},
        {NULL},
    };

    struct table_col site_cols[] = {
        {"position",
            (void *) tables->sites.position, tables->sites.num_rows, NPY_FLOAT64},
        {"ancestral_state",
            (void *) tables->sites.ancestral_state, tables->sites.ancestral_state_length,
            NPY_INT8},
        {"ancestral_state_offset",
            (void *) tables->sites.ancestral_state_offset, tables->sites.num_rows + 1,
            NPY_UINT32},
        {"metadata",
            (void *) tables->sites.metadata, tables->sites.metadata_length, NPY_INT8},
        {"metadata_offset",
            (void *) tables->sites.metadata_offset, tables->sites.num_rows + 1, NPY_UINT32},
        {NULL},
    };

    struct table_col mutation_cols[] = {
        {"site",
            (void *) tables->mutations.site, tables->mutations.num_rows, NPY_INT32},
        {"node",
            (void *) tables->mutations.node, tables->mutations.num_rows, NPY_INT32},
        {"parent",
            (void *) tables->mutations.parent, tables->mutations.num_rows, NPY_INT32},
        {"derived_state",
            (void *) tables->mutations.derived_state,
            tables->mutations.derived_state_length, NPY_INT8},
        {"derived_state_offset",
            (void *) tables->mutations.derived_state_offset,
            tables->mutations.num_rows + 1, NPY_UINT32},
        {"metadata",
            (void *) tables->mutations.metadata,
            tables->mutations.metadata_length, NPY_INT8},
        {"metadata_offset",
            (void *) tables->mutations.metadata_offset,
            tables->mutations.num_rows + 1, NPY_UINT32},
        {NULL},
    };

    struct table_col population_cols[] = {
        {"metadata", (void *) tables->populations.metadata,
            tables->populations.metadata_length, NPY_INT8},
        {"metadata_offset", (void *) tables->populations.metadata_offset,
            tables->populations.num_rows+ 1, NPY_UINT32},
        {NULL},
    };

    struct table_col provenance_cols[] = {
        {"timestamp", (void *) tables->provenances.timestamp,
            tables->provenances.timestamp_length, NPY_INT8},
        {"timestamp_offset", (void *) tables->provenances.timestamp_offset,
            tables->provenances.num_rows+ 1, NPY_UINT32},
        {"record", (void *) tables->provenances.record,
            tables->provenances.record_length, NPY_INT8},
        {"record_offset", (void *) tables->provenances.record_offset,
            tables->provenances.num_rows + 1, NPY_UINT32},
        {NULL},
    };

    struct table_desc table_descs[] = {
        {"individuals", individual_cols},
        {"nodes", node_cols},
        {"edges", edge_cols},
        {"migrations", migration_cols},
        {"sites", site_cols},
        {"mutations", mutation_cols},
        {"populations", population_cols},
        {"provenances", provenance_cols},
    };

    for (j = 0; j < sizeof(table_descs) / sizeof(*table_descs); j++) {
        table_dict = PyDict_New();
        if (table_dict == NULL) {
            goto out;
        }
        col = table_descs[j].cols;
        while (col->name != NULL) {
            array = PyArray_SimpleNewFromData(1, &col->num_rows, col->type, col->data);
            if (array == NULL) {
                goto out;
            }
            if (PyDict_SetItemString(table_dict, col->name, array) != 0) {
                goto out;
            }
            Py_DECREF(array);
            array = NULL;
            col++;
        }
        if (PyDict_SetItemString(dict, table_descs[j].name, table_dict) != 0) {
            goto out;
        }
        Py_DECREF(table_dict);
        table_dict = NULL;
    }
    ret = 0;
out:
    Py_XDECREF(array);
    Py_XDECREF(table_dict);
    return ret;
}

/* Returns a dictionary encoding of the specified table collection */
static PyObject*
dump_tables_dict(tsk_table_collection_t *tables)
{
    PyObject *ret = NULL;
    PyObject *dict = NULL;
    PyObject *val = NULL;
    int err;

    dict = PyDict_New();
    if (dict == NULL) {
        goto out;
    }
    val = Py_BuildValue("d", tables->sequence_length);
    if (val == NULL) {
        goto out;
    }
    if (PyDict_SetItemString(dict, "sequence_length", val) != 0) {
        goto out;
    }
    Py_DECREF(val);
    val = NULL;

    err = write_table_arrays(tables, dict);
    if (err != 0) {
        goto out;
    }
    ret = dict;
    dict = NULL;
out:
    Py_XDECREF(dict);
    Py_XDECREF(val);
    return ret;
}

/*===================================================================
 * LightweightTableCollection
 *===================================================================
 */

static int
LightweightTableCollection_check_state(LightweightTableCollection *self)
{
    int ret = 0;
    if (self->tables == NULL) {
        PyErr_SetString(PyExc_SystemError, "LightweightTableCollection not initialised");
        ret = -1;
    }
    return ret;
}

static void
LightweightTableCollection_dealloc(LightweightTableCollection* self)
{
    if (self->tables != NULL) {
        tsk_table_collection_free(self->tables);
        PyMem_Free(self->tables);
        self->tables = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
LightweightTableCollection_init(LightweightTableCollection *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"sequence_length", NULL};
    double sequence_length = -1;

    self->tables = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|d", kwlist, &sequence_length)) {
        goto out;
    }
    self->tables = PyMem_Malloc(sizeof(*self->tables));
    if (self->tables == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tsk_table_collection_init(self->tables, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    self->tables->sequence_length = sequence_length;
    ret = 0;
out:
    return ret;
}

static PyObject *
LightweightTableCollection_asdict(LightweightTableCollection *self)
{
    PyObject *ret = NULL;

    if (LightweightTableCollection_check_state(self) != 0) {
        goto out;
    }
    ret = dump_tables_dict(self->tables);
out:
    return ret;
}

static PyObject *
LightweightTableCollection_fromdict(LightweightTableCollection *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    PyObject *dict = NULL;

    if (!PyArg_ParseTuple(args, "O!", &PyDict_Type, &dict)) {
        goto out;
    }
    err = parse_table_collection_dict(self->tables, dict);
    if (err != 0) {
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyMemberDef LightweightTableCollection_members[] = {
    {NULL}  /* Sentinel */
};

static PyMethodDef LightweightTableCollection_methods[] = {
    {"asdict", (PyCFunction) LightweightTableCollection_asdict,
        METH_NOARGS, "Returns the tables encoded as a dictionary."},
    {"fromdict", (PyCFunction) LightweightTableCollection_fromdict,
        METH_VARARGS, "Populates the internal tables using the specified dictionary."},
    {NULL}  /* Sentinel */
};

static PyTypeObject LightweightTableCollectionType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.LightweightTableCollection",             /* tp_name */
    sizeof(LightweightTableCollection),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)LightweightTableCollection_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "LightweightTableCollection objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    LightweightTableCollection_methods,             /* tp_methods */
    LightweightTableCollection_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)LightweightTableCollection_init,      /* tp_init */
};



/*===================================================================
 * AncestorBuilder
 *===================================================================
 */

static int
AncestorBuilder_check_state(AncestorBuilder *self)
{
    int ret = 0;
    if (self->builder == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorBuilder not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorBuilder_dealloc(AncestorBuilder* self)
{
    if (self->builder != NULL) {
        ancestor_builder_free(self->builder);
        PyMem_Free(self->builder);
        self->builder = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
AncestorBuilder_init(AncestorBuilder *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"num_samples", "num_sites", NULL};
    int num_samples, num_sites;
    int flags = 0;

    self->builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                &num_samples, &num_sites)) {
        goto out;
    }
    self->builder = PyMem_Malloc(sizeof(ancestor_builder_t));
    if (self->builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_alloc(self->builder, num_samples, num_sites, flags);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
AncestorBuilder_add_site(AncestorBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"site_id", "time", "genotypes", NULL};
    PyObject *ret = NULL;
    int site_id;
    double time;
    PyObject *genotypes = NULL;
    PyArrayObject *genotypes_array = NULL;
    npy_intp *shape;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "idO!", kwlist,
            &site_id, &time, &PyArray_Type, &genotypes)) {
        goto out;
    }
    genotypes_array = (PyArrayObject *) PyArray_FROM_OTF(genotypes, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (genotypes_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(genotypes_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(genotypes_array);
    if (shape[0] != self->builder->num_samples) {
        PyErr_SetString(PyExc_ValueError, "genotypes array wrong size.");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_add_site(self->builder, (tsk_id_t) site_id, time,
            (allele_t *) PyArray_DATA(genotypes_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(genotypes_array);
    return ret;
}

static PyObject *
AncestorBuilder_make_ancestor(AncestorBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"focal_sites", "ancestor", NULL};
    PyObject *ancestor = NULL;
    PyArrayObject *ancestor_array = NULL;
    PyObject *focal_sites = NULL;
    PyArrayObject *focal_sites_array = NULL;
    size_t num_focal_sites;
    size_t num_sites;
    tsk_id_t start, end;
    npy_intp *shape;

    if (AncestorBuilder_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO!", kwlist,
            &focal_sites, &PyArray_Type, &ancestor)) {
        goto fail;
    }
    num_sites = self->builder->num_sites;
    focal_sites_array = (PyArrayObject *) PyArray_FROM_OTF(focal_sites, NPY_INT32,
            NPY_ARRAY_IN_ARRAY);
    if (focal_sites_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(focal_sites_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(focal_sites_array);
    num_focal_sites = shape[0];
    if (num_focal_sites == 0 || num_focal_sites > num_sites) {
        PyErr_SetString(PyExc_ValueError, "num_focal_sites must > 0 and <= num_sites");
        goto fail;
    }
    ancestor_array = (PyArrayObject *) PyArray_FROM_OTF(ancestor, NPY_INT8,
            NPY_ARRAY_INOUT_ARRAY);
    if (ancestor_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(ancestor_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(ancestor_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError, "input ancestor wrong size");
        goto fail;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_make_ancestor(self->builder, num_focal_sites,
        (int32_t *) PyArray_DATA(focal_sites_array),
        &start, &end, (int8_t *) PyArray_DATA(ancestor_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(focal_sites_array);
    Py_DECREF(ancestor_array);
    return Py_BuildValue("ii", start, end);
fail:
    Py_XDECREF(focal_sites_array);
    PyArray_XDECREF_ERR(ancestor_array);
    return NULL;
}

static PyObject *
AncestorBuilder_ancestor_descriptors(AncestorBuilder *self)
{
    PyObject *ret = NULL;
    PyObject *descriptors = NULL;
    PyObject *py_descriptor = NULL;
    PyArrayObject *site_array = NULL;
    ancestor_descriptor_t *descriptor;
    size_t j;
    npy_intp dims;
    int err;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    err = ancestor_builder_finalise(self->builder);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    /* ancestor_builder_print_state(self->builder, stdout); */
    descriptors = PyTuple_New(self->builder->num_ancestors);
    if (descriptors == NULL) {
        goto out;
    }
    for (j = 0; j < self->builder->num_ancestors; j++) {
        descriptor = &self->builder->descriptors[j];
        dims = descriptor->num_focal_sites;
        site_array = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32);

        if (site_array == NULL) {
            goto out;
        }
        memcpy(PyArray_DATA(site_array), descriptor->focal_sites,
                descriptor->num_focal_sites * sizeof(tsk_id_t));
        py_descriptor = Py_BuildValue("dO", descriptor->time, site_array);
        if (py_descriptor == NULL) {
            Py_DECREF(site_array);
            goto out;
        }
        PyTuple_SET_ITEM(descriptors, j, py_descriptor);
    }
    /* j = 0; */
    /* /1* It's not great that we're breaking encapsulation here and looking */
    /*  * directly in to the builder's data structures. However, it's quite an */
    /*  * awkward set of data to communicate, so it seems OK. *1/ */
    /* for (f = self->builder->num_samples; f > 0; f--) { */
    /*     for (a = self->builder->frequency_map[f].head; a != NULL; a = a->next) { */
    /*         map_elem = (pattern_map_t *) a->item; */
    /*         dims = map_elem->num_sites; */
    /*         site_array = (PyArrayObject *) PyArray_SimpleNew(1, &dims, NPY_INT32); */
    /*         if (site_array == NULL) { */
    /*             goto out; */
    /*         } */
    /*         site_array_data = (int32_t *) PyArray_DATA(site_array); */
    /*         /1* The elements are listed backwards, so reverse them *1/ */
    /*         k = map_elem->num_sites - 1; */
    /*         for (s = map_elem->sites; s != NULL; s = s->next) { */
    /*             site_array_data[k] = (int32_t) s->site; */
    /*             k--; */
    /*         } */
    /*         descriptor = Py_BuildValue("kO", (unsigned long) f, site_array); */
    /*         if (descriptor == NULL) { */
    /*             Py_DECREF(site_array); */
    /*             goto out; */
    /*         } */
    /*         PyTuple_SET_ITEM(descriptors, j, descriptor); */
    /*         j++; */
    /*     } */
    /* } */
    ret = descriptors;
    descriptors = NULL;
out:
    Py_XDECREF(descriptors);
    return ret;
}

static PyObject *
AncestorBuilder_get_num_sites(AncestorBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->builder->num_sites);
out:
    return ret;
}

static PyObject *
AncestorBuilder_get_num_ancestors(AncestorBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->builder->num_ancestors);
out:
    return ret;
}

static PyMemberDef AncestorBuilder_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef AncestorBuilder_getsetters[] = {
    {"num_sites", (getter) AncestorBuilder_get_num_sites, NULL, "The number of sites."},
    {"num_ancestors", (getter) AncestorBuilder_get_num_ancestors, NULL, "The number of ancestors."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorBuilder_methods[] = {
    {"add_site", (PyCFunction) AncestorBuilder_add_site,
        METH_VARARGS|METH_KEYWORDS,
        "Adds the specified site to this ancestor builder."},
    {"make_ancestor", (PyCFunction) AncestorBuilder_make_ancestor,
        METH_VARARGS|METH_KEYWORDS,
        "Makes the specified ancestor."},
    {"ancestor_descriptors", (PyCFunction) AncestorBuilder_ancestor_descriptors,
        METH_NOARGS,
        "Returns a list of ancestor (frequency, focal_sites) tuples."},
    {NULL}  /* Sentinel */
};

static PyTypeObject AncestorBuilderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.AncestorBuilder",             /* tp_name */
    sizeof(AncestorBuilder),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)AncestorBuilder_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "AncestorBuilder objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    AncestorBuilder_methods,             /* tp_methods */
    AncestorBuilder_members,             /* tp_members */
    AncestorBuilder_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AncestorBuilder_init,      /* tp_init */
};

/*===================================================================
 * TreeSequenceBuilder
 *===================================================================
 */

static int
TreeSequenceBuilder_check_state(TreeSequenceBuilder *self)
{
    int ret = 0;
    if (self->tree_sequence_builder == NULL) {
        PyErr_SetString(PyExc_SystemError, "TreeSequenceBuilder not initialised");
        ret = -1;
    }
    return ret;
}

static void
TreeSequenceBuilder_dealloc(TreeSequenceBuilder* self)
{
    if (self->tree_sequence_builder != NULL) {
        tree_sequence_builder_free(self->tree_sequence_builder);
        PyMem_Free(self->tree_sequence_builder);
        self->tree_sequence_builder = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
TreeSequenceBuilder_init(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    LightweightTableCollection *tables = NULL;
    PyObject *alleles = NULL;
    unsigned long max_nodes;
    unsigned long max_edges;
    static char *kwlist[] = {"tables", "alleles", "max_nodes", "max_edges", NULL};
    int flags = 0;

    self->tree_sequence_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!Okk", kwlist,
                &LightweightTableCollectionType, &tables,
                &alleles, &max_nodes, &max_edges)) {
        goto out;
    }
    self->tree_sequence_builder = PyMem_Malloc(sizeof(tree_sequence_builder_t));
    if (self->tree_sequence_builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tree_sequence_builder_alloc(self->tree_sequence_builder,
            0, max_nodes, max_edges, flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_add_node(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;

    static char *kwlist[] = {"time", "flags", NULL};
    double time;
    int flags = 1;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "d|i", kwlist, &time,
                &flags)) {
        goto out;
    }

    err = tree_sequence_builder_add_node(self->tree_sequence_builder,
            time, (uint32_t) flags);
    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", err);
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_add_path(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    int flags = 0;
    PyObject *left = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    int child;
    size_t num_edges;
    npy_intp *shape;
    int compress = 1;
    int extended_checks = 0;

    static char *kwlist[] = {"child", "left", "right", "parent",
        "compress", "extended_checks", NULL};

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kOOO|ii", kwlist,
            &child, &left, &right, &parent, &compress, &extended_checks)) {
        goto out;
    }

    if (compress) {
        flags = TSI_COMPRESS_PATH;
    }
    if (extended_checks) {
        flags |= TSI_EXTENDED_CHECKS;
    }

    /* left */
    left_array = (PyArrayObject *) PyArray_FROM_OTF(left, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (left_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(left_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(left_array);
    num_edges = shape[0];

    /* right */
    right_array = (PyArrayObject *) PyArray_FROM_OTF(right, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (right_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(right_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(right_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "right wrong size");
        goto out;
    }

    /* parent */
    parent_array = (PyArrayObject *) PyArray_FROM_OTF(parent, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (parent_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(parent_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(parent_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "parent wrong size");
        goto out;
    }

    /* WARNING!! This isn't fully safe as we're using pointers to data that can
     * be modified in Python. Must make sure that these arrays are not modified
     * by other threads. */
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_add_path(self->tree_sequence_builder,
            child, num_edges,
            (tsk_id_t *) PyArray_DATA(left_array),
            (tsk_id_t *) PyArray_DATA(right_array),
            (tsk_id_t *) PyArray_DATA(parent_array),
            flags);
    Py_END_ALLOW_THREADS

    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(parent_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_add_mutations(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *derived_state = NULL;
    PyArrayObject *derived_state_array = NULL;
    int node;
    size_t num_mutations;
    npy_intp *shape;

    static char *kwlist[] = {"node", "site", "derived_state", NULL};

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kOO", kwlist,
            &node, &site, &derived_state)) {
        goto out;
    }

    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (site_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(site_array);
    num_mutations = shape[0];

    /* derived_state */
    derived_state_array = (PyArrayObject *) PyArray_FROM_OTF(derived_state, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (derived_state_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(derived_state_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(derived_state_array);
    if (shape[0] != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "derived_state wrong size");
        goto out;
    }

    /* WARNING!! This isn't fully safe as we're using pointers to data that can
     * be modified in Python. Must make sure that these arrays are not modified
     * by other threads. */
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_add_mutations(self->tree_sequence_builder,
            node, num_mutations,
            (tsk_id_t *) PyArray_DATA(site_array),
            (allele_t *) PyArray_DATA(derived_state_array));
    Py_END_ALLOW_THREADS

    if (err < 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(site_array);
    Py_XDECREF(derived_state_array);
    return ret;
}


static PyObject *
TreeSequenceBuilder_restore_nodes(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"time", "flags", NULL};
    PyObject *time = NULL;
    PyArrayObject *time_array = NULL;
    PyObject *flags = NULL;
    PyArrayObject *flags_array = NULL;
    npy_intp *shape;
    size_t num_nodes;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &time, &flags)) {
        goto out;
    }

    /* time */
    time_array = (PyArrayObject *) PyArray_FROM_OTF(time, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (time_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(time_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(time_array);
    num_nodes = shape[0];

    /* flags */
    flags_array = (PyArrayObject *) PyArray_FROM_OTF(flags, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (flags_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(flags_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(flags_array);
    if (shape[0] != num_nodes) {
        PyErr_SetString(PyExc_ValueError, "flags array incorrect size");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_restore_nodes(self->tree_sequence_builder,
            num_nodes,
            (uint32_t *) PyArray_DATA(flags_array),
            (double *) PyArray_DATA(time_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(time_array);
    Py_XDECREF(flags_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_restore_edges(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"left", "right", "parent", "child", NULL};
    size_t num_edges;
    PyObject *left = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    PyObject *child = NULL;
    PyArrayObject *child_array = NULL;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist,
            &left, &right, &parent, &child)) {
        goto out;
    }

    /* left */
    left_array = (PyArrayObject *) PyArray_FROM_OTF(left, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (left_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(left_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(left_array);
    num_edges = shape[0];

    /* right */
    right_array = (PyArrayObject *) PyArray_FROM_OTF(right, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (right_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(right_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(right_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "right wrong size");
        goto out;
    }

    /* parent */
    parent_array = (PyArrayObject *) PyArray_FROM_OTF(parent, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (parent_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(parent_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(parent_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "parent wrong size");
        goto out;
    }

    /* child */
    child_array = (PyArrayObject *) PyArray_FROM_OTF(child, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (child_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(child_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(child_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "child wrong size");
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_restore_edges(self->tree_sequence_builder,
            num_edges,
            (tsk_id_t *) PyArray_DATA(left_array),
            (tsk_id_t *) PyArray_DATA(right_array),
            (tsk_id_t *) PyArray_DATA(parent_array),
            (tsk_id_t *) PyArray_DATA(child_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(left_array);
    Py_XDECREF(right_array);
    Py_XDECREF(parent_array);
    Py_XDECREF(child_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_restore_mutations(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"site", "node", "derived_state", "parent", NULL};
    size_t num_mutations;
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *node = NULL;
    PyArrayObject *node_array = NULL;
    PyObject *derived_state = NULL;
    PyArrayObject *derived_state_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO", kwlist,
            &site, &node, &derived_state, &parent)) {
        goto out;
    }

    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (site_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(site_array);
    num_mutations = shape[0];

    /* node */
    node_array = (PyArrayObject *) PyArray_FROM_OTF(node, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (node_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(node_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(node_array);
    if (shape[0] != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "node wrong size");
        goto out;
    }

    /* derived_state */
    derived_state_array = (PyArrayObject *) PyArray_FROM_OTF(derived_state, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (derived_state_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(derived_state_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(derived_state_array);
    if (shape[0] != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "derived_state wrong size");
        goto out;
    }

    /* parent */
    parent_array = (PyArrayObject *) PyArray_FROM_OTF(parent, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (parent_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(parent_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(parent_array);
    if (shape[0] != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "parent wrong size");
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_restore_mutations(self->tree_sequence_builder,
            num_mutations,
            (tsk_id_t *) PyArray_DATA(site_array),
            (tsk_id_t *) PyArray_DATA(node_array),
            (allele_t *) PyArray_DATA(derived_state_array));
    /* NOTE: we are ignoring parent here! */
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    Py_XDECREF(site_array);
    Py_XDECREF(node_array);
    Py_XDECREF(derived_state_array);
    Py_XDECREF(parent_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_dump(TreeSequenceBuilder *self, PyObject *args)
{
    int err;
    PyObject *ret = NULL;
    LightweightTableCollection *lwt = NULL;

    if (!PyArg_ParseTuple(args, "O!", &LightweightTableCollectionType, &lwt)) {
        goto out;
    }

    err = tree_sequence_builder_dump(self->tree_sequence_builder, lwt->tables, 0);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_freeze_indexes(TreeSequenceBuilder *self)
{
    int err;
    PyObject *ret = NULL;

    err = tree_sequence_builder_freeze_indexes(self->tree_sequence_builder);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_edges(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            tree_sequence_builder_get_num_edges(self->tree_sequence_builder));
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_nodes(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            tree_sequence_builder_get_num_nodes(self->tree_sequence_builder));
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_sites(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->tree_sequence_builder->num_sites);
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_get_num_mutations(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            tree_sequence_builder_get_num_mutations(self->tree_sequence_builder));
out:
    return ret;
}
static PyMemberDef TreeSequenceBuilder_members[] = {
    {NULL}  /* Sentinel */

};

static PyGetSetDef TreeSequenceBuilder_getsetters[] = {
    {"num_nodes", (getter) TreeSequenceBuilder_get_num_nodes, NULL,
        "The number of nodes."},
    {"num_edges", (getter) TreeSequenceBuilder_get_num_edges, NULL,
        "The number of edgess."},
    {"num_sites", (getter) TreeSequenceBuilder_get_num_sites, NULL,
        "The total number of sites."},
    {"num_mutations", (getter) TreeSequenceBuilder_get_num_mutations, NULL,
        "The total number of mutations."},
    {NULL}  /* Sentinel */
};

static PyMethodDef TreeSequenceBuilder_methods[] = {
    {"add_node", (PyCFunction) TreeSequenceBuilder_add_node,
        METH_VARARGS|METH_KEYWORDS,
        "Adds a new node to the tree sequence builder and returns its ID."},
    {"add_path", (PyCFunction) TreeSequenceBuilder_add_path,
        METH_VARARGS|METH_KEYWORDS,
        "Updates the builder with the specified copy results for a given child."},
    {"add_mutations", (PyCFunction) TreeSequenceBuilder_add_mutations,
        METH_VARARGS|METH_KEYWORDS,
        "Updates the builder with mutations for a given node."},
    {"restore_nodes", (PyCFunction) TreeSequenceBuilder_restore_nodes,
        METH_VARARGS|METH_KEYWORDS,
        "Restores the nodes in this tree sequence builder."},
    {"restore_edges", (PyCFunction) TreeSequenceBuilder_restore_edges,
        METH_VARARGS|METH_KEYWORDS,
        "Restores the edges in this tree sequence builder."},
    {"restore_mutations", (PyCFunction) TreeSequenceBuilder_restore_mutations,
        METH_VARARGS|METH_KEYWORDS,
        "Restores the mutations in this tree sequence builder."},
    {"freeze_indexes", (PyCFunction) TreeSequenceBuilder_freeze_indexes, METH_NOARGS,
        "Freezes the indexes used for ancestor matching."},
    {"dump", (PyCFunction) TreeSequenceBuilder_dump, METH_VARARGS,
        "Dumps the tree sequence data into a LightweightTableCollection"},
    {NULL}  /* Sentinel */
};

static PyTypeObject TreeSequenceBuilderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.TreeSequenceBuilder",             /* tp_name */
    sizeof(TreeSequenceBuilder),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)TreeSequenceBuilder_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "TreeSequenceBuilder objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    TreeSequenceBuilder_methods,             /* tp_methods */
    TreeSequenceBuilder_members,             /* tp_members */
    TreeSequenceBuilder_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)TreeSequenceBuilder_init,      /* tp_init */
};


/*===================================================================
 * AncestorMatcher
 *===================================================================
 */

static int
AncestorMatcher_check_state(AncestorMatcher *self)
{
    int ret = 0;
    if (self->ancestor_matcher == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorMatcher not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorMatcher_dealloc(AncestorMatcher* self)
{
    if (self->ancestor_matcher != NULL) {
        ancestor_matcher_free(self->ancestor_matcher);
        PyMem_Free(self->ancestor_matcher);
        self->ancestor_matcher = NULL;
    }
    Py_XDECREF(self->tree_sequence_builder);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
AncestorMatcher_init(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    int extended_checks = 0;
    static char *kwlist[] = {"tree_sequence_builder", "recombination_rate",
        "mutation_rate", "precision", "extended_checks", NULL};
    TreeSequenceBuilder *tree_sequence_builder = NULL;
    PyObject *recombination_rate = NULL;
    PyObject *mutation_rate = NULL;
    PyArrayObject *recombination_rate_array = NULL;
    PyArrayObject *mutation_rate_array = NULL;
    npy_intp *shape;
    unsigned int precision = 22;
    int flags = 0;

    self->ancestor_matcher = NULL;
    self->tree_sequence_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!OO|Ii", kwlist,
                &TreeSequenceBuilderType, &tree_sequence_builder,
                &recombination_rate, &mutation_rate, &precision,
                &extended_checks)) {
        goto out;
    }
    self->tree_sequence_builder = tree_sequence_builder;
    Py_INCREF(self->tree_sequence_builder);
    if (TreeSequenceBuilder_check_state(self->tree_sequence_builder) != 0) {
        goto out;
    }

    recombination_rate_array = (PyArrayObject *) PyArray_FromAny(recombination_rate,
            PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
            NPY_ARRAY_IN_ARRAY, NULL);
    if (recombination_rate_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(recombination_rate_array);
    if (shape[0] != tree_sequence_builder->tree_sequence_builder->num_sites) {
        PyErr_SetString(PyExc_ValueError,
                "Size of recombination_rate array must be num_sites");
        goto out;
    }
    mutation_rate_array = (PyArrayObject *) PyArray_FromAny(mutation_rate,
            PyArray_DescrFromType(NPY_FLOAT64), 1, 1,
            NPY_ARRAY_IN_ARRAY, NULL);
    if (mutation_rate_array == NULL) {
        goto out;
    }
    shape = PyArray_DIMS(mutation_rate_array);
    if (shape[0] != tree_sequence_builder->tree_sequence_builder->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Size of mutation_rate array must be num_sites");
        goto out;
    }

    self->ancestor_matcher = PyMem_Malloc(sizeof(ancestor_matcher_t));
    if (self->ancestor_matcher == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    if (extended_checks) {
        flags = TSI_EXTENDED_CHECKS;
    }
    err = ancestor_matcher_alloc(self->ancestor_matcher,
            self->tree_sequence_builder->tree_sequence_builder,
            PyArray_DATA(recombination_rate_array),
            PyArray_DATA(mutation_rate_array),
            precision, flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(recombination_rate_array);
    Py_XDECREF(mutation_rate_array);
    return ret;
}

static PyObject *
AncestorMatcher_find_path(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"haplotype", "start", "end", "match", NULL};
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    PyObject *match = NULL;
    PyArrayObject *match_array = NULL;
    npy_intp *shape;
    size_t num_edges;
    int start, end;
    tsk_id_t *ret_left, *ret_right;
    tsk_id_t *ret_parent;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    npy_intp dims[1];

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OiiO!", kwlist,
                &haplotype, &start, &end, &PyArray_Type, &match)) {
        goto out;
    }
    haplotype_array = (PyArrayObject *) PyArray_FROM_OTF(haplotype, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (haplotype_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(haplotype_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect size for input haplotype.");
        goto out;
    }

    match_array = (PyArrayObject *) PyArray_FROM_OTF(match, NPY_INT8,
            NPY_ARRAY_INOUT_ARRAY);
    if (match_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(match_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(match_array);
    if (shape[0] != self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "input match wrong size");
        goto out;
    }

    Py_BEGIN_ALLOW_THREADS
    err = ancestor_matcher_find_path(self->ancestor_matcher,
            (tsk_id_t) start, (tsk_id_t) end, (allele_t *) PyArray_DATA(haplotype_array),
            (allele_t *) PyArray_DATA(match_array),
            &num_edges, &ret_left, &ret_right, &ret_parent);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = num_edges;
    left = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT32);
    right = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_UINT32);
    parent = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_INT32);
    if (left == NULL || right == NULL || parent == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(left), ret_left, num_edges * sizeof(*ret_left));
    memcpy(PyArray_DATA(right), ret_right, num_edges * sizeof(*ret_right));
    memcpy(PyArray_DATA(parent), ret_parent, num_edges * sizeof(*ret_parent));
    ret = Py_BuildValue("(OOO)", left, right, parent);
    if (ret == NULL) {
        goto out;
    }
    left = NULL;
    right = NULL;
    parent = NULL;
out:
    Py_XDECREF(haplotype_array);
    Py_XDECREF(match_array);
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    return ret;
}

static PyObject *
AncestorMatcher_get_traceback(AncestorMatcher *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned long site;
    node_state_list_t *list;
    PyObject *dict = NULL;
    PyObject *key = NULL;
    PyObject *value = NULL;
    int j;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTuple(args, "k", &site)) {
        goto out;
    }
    if (site >= self->ancestor_matcher->num_sites) {
        PyErr_SetString(PyExc_ValueError, "site out of range");
        goto out;
    }
    dict = PyDict_New();
    if (dict == NULL) {
        goto out;
    }
    list = &self->ancestor_matcher->traceback[site];
    for (j = 0; j < list->size; j++) {
        key = Py_BuildValue("k", (unsigned long) list->node[j]);
        value = Py_BuildValue("i", (int) list->recombination_required[j]);
        if (key == NULL || value == NULL) {
            goto out;
        }
        if (PyDict_SetItem(dict, key, value) != 0) {
            goto out;
        }
        Py_DECREF(key);
        key = NULL;
        Py_DECREF(value);
        value = NULL;
    }
    ret = dict;
    dict = NULL;
out:
    Py_XDECREF(key);
    Py_XDECREF(value);
    Py_XDECREF(dict);
    return ret;
}

static PyObject *
AncestorMatcher_get_mean_traceback_size(AncestorMatcher *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("d", ancestor_matcher_get_mean_traceback_size(
                self->ancestor_matcher));
out:
    return ret;
}

static PyObject *
AncestorMatcher_get_total_memory(AncestorMatcher *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long)
            ancestor_matcher_get_total_memory(self->ancestor_matcher));
out:
    return ret;
}


static PyMemberDef AncestorMatcher_members[] = {
    {NULL}  /* Sentinel */

};

static PyGetSetDef AncestorMatcher_getsetters[] = {
    {"mean_traceback_size", (getter) AncestorMatcher_get_mean_traceback_size,
        NULL, "The mean size of the traceback per site."},
    {"total_memory", (getter) AncestorMatcher_get_total_memory,
        NULL, "The total amount of memory used by this matcher."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorMatcher_methods[] = {
    {"find_path", (PyCFunction) AncestorMatcher_find_path,
        METH_VARARGS|METH_KEYWORDS,
        "Returns a best match path for the specified haplotype through the ancestors."},
    {"get_traceback", (PyCFunction) AncestorMatcher_get_traceback,
        METH_VARARGS, "Returns the traceback likelihood dictionary at the specified site."},
    {NULL}  /* Sentinel */
};

static PyTypeObject AncestorMatcherType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.AncestorMatcher",             /* tp_name */
    sizeof(AncestorMatcher),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)AncestorMatcher_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "AncestorMatcher objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    AncestorMatcher_methods,             /* tp_methods */
    AncestorMatcher_members,             /* tp_members */
    AncestorMatcher_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AncestorMatcher_init,      /* tp_init */
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyMethodDef tsinfer_methods[] = {
    {NULL}        /* Sentinel */
};

/* Initialisation code supports Python 2.x and 3.x. The framework uses the
 * recommended structure from http://docs.python.org/howto/cporting.html.
 * I've ignored the point about storing state in globals, as the examples
 * from the Python documentation still use this idiom.
 */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef tsinfermodule = {
    PyModuleDef_HEAD_INIT,
    "_tsinfer",   /* name of module */
    MODULE_DOC, /* module documentation, may be NULL */
    -1,
    tsinfer_methods,
    NULL, NULL, NULL, NULL
};

#define INITERROR return NULL

PyObject *
PyInit__tsinfer(void)

#else
#define INITERROR return

void
init_tsinfer(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&tsinfermodule);
#else
    PyObject *module = Py_InitModule3("_tsinfer", tsinfer_methods, MODULE_DOC);
#endif
    if (module == NULL) {
        INITERROR;
    }
    /* Initialise numpy */
    import_array();

    /* LightweightTableCollection type */
    LightweightTableCollectionType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&LightweightTableCollectionType) < 0) {
        return NULL;;
    }
    Py_INCREF(&LightweightTableCollectionType);
    PyModule_AddObject(module, "LightweightTableCollection",
            (PyObject *) &LightweightTableCollectionType);

    /* AncestorBuilder type */
    AncestorBuilderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AncestorBuilderType) < 0) {
        INITERROR;
    }
    Py_INCREF(&AncestorBuilderType);
    PyModule_AddObject(module, "AncestorBuilder", (PyObject *) &AncestorBuilderType);
    /* AncestorMatcher type */
    AncestorMatcherType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AncestorMatcherType) < 0) {
        INITERROR;
    }
    Py_INCREF(&AncestorMatcherType);
    PyModule_AddObject(module, "AncestorMatcher", (PyObject *) &AncestorMatcherType);
    /* TreeSequenceBuilder type */
    TreeSequenceBuilderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&TreeSequenceBuilderType) < 0) {
        INITERROR;
    }
    Py_INCREF(&TreeSequenceBuilderType);
    PyModule_AddObject(module, "TreeSequenceBuilder", (PyObject *) &TreeSequenceBuilderType);

    TsinfLibraryError = PyErr_NewException("_tsinfer.LibraryError", NULL, NULL);
    Py_INCREF(TsinfLibraryError);
    PyModule_AddObject(module, "LibraryError", TsinfLibraryError);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
