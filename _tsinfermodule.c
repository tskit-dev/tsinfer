
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
    PyErr_Format(TsinfLibraryError, "Error occured: %d", err);
}

/* TODO change this to return a numpy array. */
static PyObject *
convert_site_id_list(site_id_t *sites, size_t num_sites)
{
    PyObject *ret = NULL;
    PyObject *t;
    PyObject *py_int;
    size_t j;

    t = PyTuple_New(num_sites);
    if (t == NULL) {
        goto out;
    }
    for (j = 0; j < num_sites; j++) {
        py_int = Py_BuildValue("k", (unsigned long) sites[j]);
        if (py_int == NULL) {
            Py_DECREF(t);
            goto out;
        }
        PyTuple_SET_ITEM(t, j, py_int);
    }
    ret = t;
out:
    return ret;
}

/* static PyObject * */
/* convert_segment_list(segment_list_t *list) */
/* { */
/*     PyObject *ret = NULL; */
/*     PyObject *t; */
/*     PyObject *py_int; */
/*     size_t j; */
/*     segment_list_node_t *u; */

/*     t = PyTuple_New(list->length); */
/*     if (t == NULL) { */
/*         goto out; */
/*     } */
/*     j = 0; */
/*     for (u = list->head; u != NULL; u = u->next) { */
/*         py_int = Py_BuildValue("(k,k)", (unsigned long) u->start, */
/*                 (unsigned long) u->end); */
/*         if (py_int == NULL) { */
/*             Py_DECREF(t); */
/*             goto out; */
/*         } */
/*         PyTuple_SET_ITEM(t, j, py_int); */
/*         j++; */
/*     } */
/*     ret = t; */
/* out: */
/*     return ret; */
/* } */



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
    int err;
    static char *kwlist[] = {"samples", "position", NULL};
    size_t num_samples, num_sites;
    PyObject *samples = NULL;
    PyArrayObject *samples_array = NULL;
    PyObject *position = NULL;
    PyArrayObject *position_array = NULL;
    npy_intp *shape;

    self->builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO", kwlist, &samples, &position)) {
        goto fail;
    }
    samples_array = (PyArrayObject *) PyArray_FROM_OTF(samples, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (samples_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(samples_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Dim != 2");
        goto fail;
    }
    shape = PyArray_DIMS(samples_array);
    num_samples = shape[0];
    num_sites = shape[1];
    if (num_samples < 2) {
        PyErr_SetString(PyExc_ValueError, "Need > 2 samples");
        goto fail;
    }
    if (num_sites < 1) {
        PyErr_SetString(PyExc_ValueError, "Must have > 0 sites");
        goto fail;
    }
    position_array = (PyArrayObject *) PyArray_FROM_OTF(position, NPY_FLOAT64,
            NPY_ARRAY_IN_ARRAY);
    if (position_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(position_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(position_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError, "position num sites mismatch");
        goto fail;
    }

    self->builder = PyMem_Malloc(sizeof(ancestor_builder_t));
    if (self->builder == NULL) {
        PyErr_NoMemory();
        goto fail;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_alloc(self->builder, num_samples, num_sites,
            (double *) PyArray_DATA(position_array),
            (int8_t *) PyArray_DATA(samples_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(samples_array);
    return 0;
fail:
    PyArray_XDECREF_ERR(samples_array);
    return -1;
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
    npy_intp *shape;

    if (AncestorBuilder_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO!", kwlist,
            &focal_sites, &PyArray_Type, &ancestor)) {
        goto fail;
    }
    num_sites = self->builder->num_sites;
    focal_sites_array = (PyArrayObject *) PyArray_FROM_OTF(focal_sites, NPY_UINT32,
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
        (uint32_t *) PyArray_DATA(focal_sites_array),
        (int8_t *) PyArray_DATA(ancestor_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(focal_sites_array);
    Py_DECREF(ancestor_array);
    return Py_BuildValue("");
fail:
    Py_XDECREF(focal_sites_array);
    PyArray_XDECREF_ERR(ancestor_array);
    return NULL;
}

static PyObject *
convert_ancestor_focal_sites(frequency_class_t *class)
{
    PyObject *ret = NULL;
    PyObject *py_ancestor_list = NULL;
    PyObject *py_ancestor = NULL;
    size_t j;

    py_ancestor_list = PyTuple_New(class->num_ancestors);
    if (py_ancestor_list == NULL) {
        goto out;
    }
    for (j = 0; j < class->num_ancestors; j++) {
        py_ancestor = convert_site_id_list(
                class->ancestor_focal_sites[j], class->num_ancestor_focal_sites[j]);
        if (py_ancestor == NULL) {
            Py_DECREF(py_ancestor_list);
            goto out;
        }
        PyTuple_SET_ITEM(py_ancestor_list, j, py_ancestor);
    }
    ret = py_ancestor_list;
out:
    return ret;
}

static PyObject *
AncestorBuilder_get_frequency_classes(AncestorBuilder *self)
{
    PyObject *ret = NULL;
    PyObject *py_classes = NULL;
    PyObject *py_class = NULL;
    PyObject *py_sites_lists = NULL;
    frequency_class_t *class;
    size_t j;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }

    py_classes = PyTuple_New(self->builder->num_frequency_classes);
    if (py_classes == NULL) {
        goto out;
    }
    for (j = 0; j < self->builder->num_frequency_classes; j++) {
        class = self->builder->frequency_classes + j;
        py_sites_lists = convert_ancestor_focal_sites(class);
        if (py_sites_lists == NULL) {
            Py_DECREF(py_classes);
            goto out;
        }
        py_class = Py_BuildValue("kO", (unsigned long) class->frequency, py_sites_lists);
        if (py_class == NULL) {
            Py_DECREF(py_sites_lists);
            Py_DECREF(py_classes);
            goto out;
        }
        PyTuple_SET_ITEM(py_classes, j, py_class);
    }
    ret = py_classes;
out:
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
    {"make_ancestor", (PyCFunction) AncestorBuilder_make_ancestor,
        METH_VARARGS|METH_KEYWORDS,
        "Makes the specified ancestor."},
    {"get_frequency_classes", (PyCFunction) AncestorBuilder_get_frequency_classes,
        METH_NOARGS,
        "Returns a list of (frequency, sites) tuples"},
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
    unsigned long num_sites;
    unsigned long max_nodes;
    unsigned long max_edges;
    int resolve_shared_recombs = 1;
    static char *kwlist[] = {"num_sites", "max_nodes", "max_edges",
        "resolve_shared_recombinations", NULL};
    int flags = 0;

    self->tree_sequence_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kkk|i", kwlist,
                &num_sites, &max_nodes, &max_edges, &resolve_shared_recombs)) {
        goto out;
    }
    self->tree_sequence_builder = PyMem_Malloc(sizeof(tree_sequence_builder_t));
    if (self->tree_sequence_builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    if (resolve_shared_recombs) {
        flags |= TSI_RESOLVE_SHARED_RECOMBS;
    }

    err = tree_sequence_builder_alloc(self->tree_sequence_builder, num_sites,
            max_nodes, max_edges, flags);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
TreeSequenceBuilder_update(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"num_nodes", "time",
        /*edgesets */
        "left", "right", "parent", "child",
        /* mutations */
        "site", "node", NULL};
    unsigned long num_nodes;
    double time;
    size_t num_edges, num_mutations;
    PyObject *left = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    PyObject *child = NULL;
    PyArrayObject *child_array = NULL;
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *node = NULL;
    PyArrayObject *node_array = NULL;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kdOOOOOO", kwlist,
            &num_nodes, &time,
            &left, &right, &parent, &child,
            &site, &node)) {
        goto out;
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

    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_UINT32, NPY_ARRAY_IN_ARRAY);
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
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_update(self->tree_sequence_builder,
            num_nodes, time, num_edges,
            (site_id_t *) PyArray_DATA(left_array),
            (site_id_t *) PyArray_DATA(right_array),
            (node_id_t *) PyArray_DATA(parent_array),
            (node_id_t *) PyArray_DATA(child_array),
            num_mutations,
            (site_id_t *) PyArray_DATA(site_array),
            (node_id_t *) PyArray_DATA(node_array));
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
    Py_XDECREF(site_array);
    Py_XDECREF(node_array);
    return ret;
}

static PyObject *
TreeSequenceBuilder_dump_nodes(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"flags", "time", NULL};
    PyObject *time = NULL;
    PyArrayObject *time_array = NULL;
    PyObject *flags = NULL;
    PyArrayObject *flags_array = NULL;
    size_t num_nodes;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!", kwlist,
            &PyArray_Type, &flags, &PyArray_Type, &time)) {
        goto fail;
    }
    num_nodes = self->tree_sequence_builder->num_nodes;
    /* time */
    time_array = (PyArrayObject *) PyArray_FROM_OTF(time, NPY_FLOAT64,
            NPY_ARRAY_INOUT_ARRAY);
    if (time_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(time_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(time_array);
    if (shape[0] != num_nodes) {
        PyErr_SetString(PyExc_ValueError, "input time wrong size");
        goto fail;
    }
    /* flags */
    flags_array = (PyArrayObject *) PyArray_FROM_OTF(flags, NPY_UINT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (flags_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(flags_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(flags_array);
    if (shape[0] != num_nodes) {
        PyErr_SetString(PyExc_ValueError, "input flags wrong size");
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_dump_nodes(self->tree_sequence_builder,
        (uint32_t *) PyArray_DATA(flags_array),
        (double *) PyArray_DATA(time_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(time_array);
    Py_DECREF(flags_array);
    return Py_BuildValue("");
fail:
    PyArray_XDECREF_ERR(time_array);
    PyArray_XDECREF_ERR(flags_array);
    return NULL;
}

static PyObject *
TreeSequenceBuilder_dump_edges(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"left", "right", "parent", "child", NULL};
    PyObject *left = NULL;
    PyArrayObject *left_array = NULL;
    PyObject *right = NULL;
    PyArrayObject *right_array = NULL;
    PyObject *parent = NULL;
    PyArrayObject *parent_array = NULL;
    PyObject *child = NULL;
    PyArrayObject *child_array = NULL;
    size_t num_edges;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!O!", kwlist,
            &PyArray_Type, &left,
            &PyArray_Type, &right,
            &PyArray_Type, &parent,
            &PyArray_Type, &child)) {
        goto fail;
    }
    num_edges = self->tree_sequence_builder->num_edges;
    /* left */
    left_array = (PyArrayObject *) PyArray_FROM_OTF(left, NPY_FLOAT64,
            NPY_ARRAY_INOUT_ARRAY);
    if (left_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(left_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(left_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "input left wrong size");
        goto fail;
    }
    /* right */
    right_array = (PyArrayObject *) PyArray_FROM_OTF(right, NPY_FLOAT64,
            NPY_ARRAY_INOUT_ARRAY);
    if (right_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(right_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(right_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "input right wrong size");
        goto fail;
    }
    /* parent */
    parent_array = (PyArrayObject *) PyArray_FROM_OTF(parent, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (parent_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(parent_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(parent_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "input parent wrong size");
        goto fail;
    }
    /* child */
    child_array = (PyArrayObject *) PyArray_FROM_OTF(child, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (child_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(child_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(child_array);
    if (shape[0] != num_edges) {
        PyErr_SetString(PyExc_ValueError, "input child wrong size");
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_dump_edges(self->tree_sequence_builder,
        (double *) PyArray_DATA(left_array),
        (double *) PyArray_DATA(right_array),
        (ancestor_id_t *) PyArray_DATA(parent_array),
        (ancestor_id_t *) PyArray_DATA(child_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(left_array);
    Py_DECREF(right_array);
    Py_DECREF(parent_array);
    Py_DECREF(child_array);
    return Py_BuildValue("");
fail:
    PyArray_XDECREF_ERR(left_array);
    PyArray_XDECREF_ERR(right_array);
    PyArray_XDECREF_ERR(parent_array);
    PyArray_XDECREF_ERR(child_array);
    return NULL;
}

static PyObject *
TreeSequenceBuilder_dump_mutations(TreeSequenceBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"site", "node", "derived_state", NULL};
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *node = NULL;
    PyArrayObject *node_array = NULL;
    PyObject *derived_state = NULL;
    PyArrayObject *derived_state_array = NULL;
    size_t num_mutations;
    npy_intp *shape;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!", kwlist,
            &PyArray_Type, &site,
            &PyArray_Type, &node,
            &PyArray_Type, &derived_state)) {
        goto fail;
    }
    num_mutations = self->tree_sequence_builder->num_mutations;
    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (site_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(site_array);
    if (shape[0] != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "input site wrong size");
        goto fail;
    }
    /* node */
    node_array = (PyArrayObject *) PyArray_FROM_OTF(node, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (node_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(node_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(node_array);
    if (shape[0] != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "input node wrong size");
        goto fail;
    }
    /* derived_state */
    derived_state_array = (PyArrayObject *) PyArray_FROM_OTF(derived_state, NPY_INT8,
            NPY_ARRAY_INOUT_ARRAY);
    if (derived_state_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(derived_state_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(derived_state_array);
    if (shape[0] != num_mutations) {
        PyErr_SetString(PyExc_ValueError, "input derived_state wrong size");
        goto fail;
    }

    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_dump_mutations(self->tree_sequence_builder,
        (site_id_t *) PyArray_DATA(site_array),
        (ancestor_id_t *) PyArray_DATA(node_array),
        (allele_t *) PyArray_DATA(derived_state_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(site_array);
    Py_DECREF(node_array);
    Py_DECREF(derived_state_array);
    return Py_BuildValue("");
fail:
    PyArray_XDECREF_ERR(site_array);
    PyArray_XDECREF_ERR(node_array);
    PyArray_XDECREF_ERR(derived_state_array);
    return NULL;
}

static PyObject *
TreeSequenceBuilder_get_num_edges(TreeSequenceBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (TreeSequenceBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->tree_sequence_builder->num_edges);
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
    ret = Py_BuildValue("k", (unsigned long) self->tree_sequence_builder->num_nodes);
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
    ret = Py_BuildValue("k", (unsigned long) self->tree_sequence_builder->num_mutations);
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
    {"update", (PyCFunction) TreeSequenceBuilder_update,
        METH_VARARGS|METH_KEYWORDS,
        "Updates the builder with the specified copy results."},
    {"dump_nodes", (PyCFunction) TreeSequenceBuilder_dump_nodes,
        METH_VARARGS|METH_KEYWORDS,
        "Dumps node data into numpy arrays."},
    {"dump_edges", (PyCFunction) TreeSequenceBuilder_dump_edges,
        METH_VARARGS|METH_KEYWORDS,
        "Dumps edgeset data into numpy arrays."},
    {"dump_mutations", (PyCFunction) TreeSequenceBuilder_dump_mutations,
        METH_VARARGS|METH_KEYWORDS,
        "Dumps mutation data into numpy arrays."},
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
    static char *kwlist[] = {"tree_sequence_builder", "recombination_rate", NULL};
    TreeSequenceBuilder *tree_sequence_builder = NULL;
    double recombination_rate;

    self->ancestor_matcher = NULL;
    self->tree_sequence_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!d", kwlist,
                &TreeSequenceBuilderType, &tree_sequence_builder,
                &recombination_rate)) {
        goto out;
    }
    self->tree_sequence_builder = tree_sequence_builder;
    Py_INCREF(self->tree_sequence_builder);
    if (TreeSequenceBuilder_check_state(self->tree_sequence_builder) != 0) {
        goto out;
    }
    self->ancestor_matcher = PyMem_Malloc(sizeof(ancestor_matcher_t));
    if (self->ancestor_matcher == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = ancestor_matcher_alloc(self->ancestor_matcher,
            self->tree_sequence_builder->tree_sequence_builder, recombination_rate);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
AncestorMatcher_find_path(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"haplotype", NULL};
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape;
    size_t num_edges, num_mismatches;
    site_id_t *ret_left, *ret_right, *ret_mismatches;
    node_id_t *ret_parent;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    PyArrayObject *mismatches = NULL;
    npy_intp dims[1];

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &haplotype)) {
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
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_matcher_find_path(self->ancestor_matcher,
            (allele_t *) PyArray_DATA(haplotype_array),
            &num_edges, &ret_left, &ret_right, &ret_parent,
            &num_mismatches, &ret_mismatches);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = num_edges;
    left = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_UINT32, ret_left);
    right = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_UINT32, ret_right);
    parent = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_INT32, ret_parent);
    dims[0] = num_mismatches;
    mismatches = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, NPY_UINT32,
            ret_mismatches);
    if (left == NULL || right == NULL || parent == NULL || mismatches == NULL) {
        goto out;
    }
    ret = Py_BuildValue("(OOO)O", left, right, parent, mismatches);
out:
    Py_XDECREF(haplotype_array);
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    Py_XDECREF(mismatches);
    return ret;
}

static PyObject *
AncestorMatcher_get_traceback(AncestorMatcher *self, PyObject *args)
{
    PyObject *ret = NULL;
    unsigned long site;
    likelihood_list_t *z;
    PyObject *dict = NULL;
    PyObject *key = NULL;
    PyObject *value = NULL;

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
    for (z = self->ancestor_matcher->traceback[site]; z != NULL; z = z->next) {
        key = Py_BuildValue("k", (unsigned long) z->node);
        value = Py_BuildValue("d", z->likelihood);
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
