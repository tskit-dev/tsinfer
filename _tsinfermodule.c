
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <float.h>
#include <stdbool.h>

#include "lib/tsinfer.h"
// deprecated
#include "lib/ls.h"

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
    ancestor_store_t *store;
} AncestorStore;

typedef struct {
    PyObject_HEAD
    ancestor_store_builder_t *store_builder;
} AncestorStoreBuilder;


typedef struct {
    PyObject_HEAD
    tree_sequence_builder_t *tree_sequence_builder;
} TreeSequenceBuilder;

typedef struct {
    PyObject_HEAD
    ancestor_matcher_t *ancestor_matcher;
    TreeSequenceBuilder *tree_sequence_builder;
} AncestorMatcher;

/* Deprecated */
typedef struct {
    PyObject_HEAD
    reference_panel_t *reference_panel;
    double sequence_length;
    unsigned int num_haplotypes;
    unsigned int num_sites;
    unsigned int num_samples;
} ReferencePanel;

typedef struct {
    PyObject_HEAD
    ReferencePanel *reference_panel;
    threader_t *threader;
} Threader;

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

static PyObject *
convert_site(site_state_t *site)
{
    PyObject *ret = NULL;
    PyObject *t;
    PyObject *item;
    size_t j;

    t = PyTuple_New(site->num_segments);
    if (t == NULL) {
        goto out;
    }
    for (j = 0; j < site->num_segments; j++) {
        item = Py_BuildValue("kk", (unsigned long) site->start[j],
                (unsigned long) site->end[j]);
        if (item == NULL) {
            Py_DECREF(t);
            goto out;
        }
        PyTuple_SET_ITEM(t, j, item);
    }
    ret = t;
out:
    return ret;
}



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
 * AncestorStoreBuilder
 *===================================================================
 */

static int
AncestorStoreBuilder_check_state(AncestorStoreBuilder *self)
{
    int ret = 0;
    if (self->store_builder == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorStoreBuilder not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorStoreBuilder_dealloc(AncestorStoreBuilder* self)
{
    if (self->store_builder != NULL) {
        ancestor_store_builder_free(self->store_builder);
        PyMem_Free(self->store_builder);
        self->store_builder = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
AncestorStoreBuilder_init(AncestorStoreBuilder *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"num_sites", "segment_block_size", NULL};
    unsigned long num_sites;
    unsigned long segment_block_size = 1024 * 1024;

    self->store_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k|k", kwlist, &num_sites,
                &segment_block_size)) {
        goto out;
    }
    if (num_sites < 1) {
        PyErr_SetString(PyExc_ValueError, "Must have > 0 sites");
        goto out;
    }
    if (segment_block_size < 1) {
        PyErr_SetString(PyExc_ValueError, "Must have > 0 block size");
        goto out;
    }
    self->store_builder = PyMem_Malloc(sizeof(ancestor_store_builder_t));
    if (self->store_builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = ancestor_store_builder_alloc(self->store_builder, num_sites, segment_block_size);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
AncestorStoreBuilder_add(AncestorStoreBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"ancestor", NULL};
    PyObject *ancestor = NULL;
    PyArrayObject *ancestor_array = NULL;
    npy_intp *shape;

    if (AncestorStoreBuilder_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &ancestor)) {
        goto fail;
    }
    ancestor_array = (PyArrayObject *) PyArray_FROM_OTF(ancestor, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (ancestor_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(ancestor_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(ancestor_array);
    if (shape[0] != self->store_builder->num_sites) {
        PyErr_SetString(PyExc_ValueError, "input ancestor wrong size");
        goto fail;
    }
    err = ancestor_store_builder_add(self->store_builder, (int8_t *) PyArray_DATA(ancestor_array));
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(ancestor_array);
    Py_INCREF(Py_None);
    return Py_None;
fail:
    Py_XDECREF(ancestor_array);
    return NULL;
}

static PyObject *
AncestorStoreBuilder_dump_segments(AncestorStoreBuilder *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"site", "start", "end", NULL};
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *start = NULL;
    PyArrayObject *start_array = NULL;
    PyObject *end = NULL;
    PyArrayObject *end_array = NULL;
    size_t total_segments;
    npy_intp *shape;

    if (AncestorStoreBuilder_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!O!O!", kwlist,
            &PyArray_Type, &site, &PyArray_Type, &start, &PyArray_Type, &end)) {
        goto fail;
    }
    total_segments = self->store_builder->total_segments;

    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_UINT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (site_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(site_array);
    if (shape[0] != total_segments) {
        PyErr_SetString(PyExc_ValueError, "input site wrong size");
        goto fail;
    }
    /* start */
    start_array = (PyArrayObject *) PyArray_FROM_OTF(start, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (start_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(start_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(start_array);
    if (shape[0] != total_segments) {
        PyErr_SetString(PyExc_ValueError, "input start wrong size");
        goto fail;
    }
    /* end */
    end_array = (PyArrayObject *) PyArray_FROM_OTF(end, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (end_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(end_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(end_array);
    if (shape[0] != total_segments) {
        PyErr_SetString(PyExc_ValueError, "input end wrong size");
        goto fail;
    }

    err = ancestor_store_builder_dump(self->store_builder,
        (uint32_t *) PyArray_DATA(site_array),
        (int32_t *) PyArray_DATA(start_array),
        (int32_t *) PyArray_DATA(end_array));
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(site_array);
    Py_DECREF(start_array);
    Py_DECREF(end_array);
    return Py_BuildValue("");
fail:
    PyArray_XDECREF_ERR(site_array);
    PyArray_XDECREF_ERR(start_array);
    PyArray_XDECREF_ERR(end_array);
    return NULL;
}

static PyObject *
AncestorStoreBuilder_get_num_sites(AncestorStoreBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStoreBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store_builder->num_sites);
out:
    return ret;
}

static PyObject *
AncestorStoreBuilder_get_num_ancestors(AncestorStoreBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStoreBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store_builder->num_ancestors);
out:
    return ret;
}

static PyObject *
AncestorStoreBuilder_get_total_segments(AncestorStoreBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStoreBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store_builder->total_segments);
out:
    return ret;
}

static PyMemberDef AncestorStoreBuilder_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef AncestorStoreBuilder_getsetters[] = {
    {"num_sites", (getter) AncestorStoreBuilder_get_num_sites, NULL, "The number of sites."},
    {"num_ancestors", (getter) AncestorStoreBuilder_get_num_ancestors, NULL, "The number of ancestors."},
    {"total_segments", (getter) AncestorStoreBuilder_get_total_segments, NULL,
        "The total number of segments across all sites."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorStoreBuilder_methods[] = {
    {"add", (PyCFunction) AncestorStoreBuilder_add,
        METH_VARARGS|METH_KEYWORDS,
        "Adds the specified ancestor."},
    {"dump_segments", (PyCFunction) AncestorStoreBuilder_dump_segments,
        METH_VARARGS|METH_KEYWORDS,
        "Dumps all segments into the specified numpy arrays."},
    {NULL}  /* Sentinel */
};

static PyTypeObject AncestorStoreBuilderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.AncestorStoreBuilder",             /* tp_name */
    sizeof(AncestorStoreBuilder),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)AncestorStoreBuilder_dealloc, /* tp_dealloc */
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
    "AncestorStoreBuilder objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    AncestorStoreBuilder_methods,             /* tp_methods */
    AncestorStoreBuilder_members,             /* tp_members */
    AncestorStoreBuilder_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AncestorStoreBuilder_init,      /* tp_init */
};

/*===================================================================
 * AncestorStore
 *===================================================================
 */

static int
AncestorStore_check_state(AncestorStore *self)
{
    int ret = 0;
    if (self->store == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorStore not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorStore_dealloc(AncestorStore* self)
{
    if (self->store != NULL) {
        ancestor_store_free(self->store);
        PyMem_Free(self->store);
        self->store = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
AncestorStore_init(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"position", "ancestor_age",
        "focal_site_ancestor", "focal_site",
        "site", "start", "end", NULL};
    PyObject *position = NULL;
    PyArrayObject *position_array = NULL;
    PyObject *ancestor_age = NULL;
    PyArrayObject *ancestor_age_array = NULL;
    PyObject *focal_site = NULL;
    PyArrayObject *focal_site_array = NULL;
    PyObject *focal_site_ancestor = NULL;
    PyArrayObject *focal_site_ancestor_array = NULL;
    PyObject *site = NULL;
    PyArrayObject *site_array = NULL;
    PyObject *start = NULL;
    PyArrayObject *start_array = NULL;
    PyObject *end = NULL;
    PyArrayObject *end_array = NULL;
    size_t num_sites, num_ancestors, num_focal_sites, total_segments;
    npy_intp *shape;

    self->store = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOOOOO", kwlist,
            &position, &ancestor_age,
            &focal_site_ancestor, &focal_site,
            &site, &start, &end)) {
        goto out;
    }
    /* position */
    position_array = (PyArrayObject *) PyArray_FROM_OTF(position, NPY_FLOAT64,
            NPY_ARRAY_IN_ARRAY);
    if (position_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(position_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(position_array);
    num_sites = shape[0];
    if (num_sites < 1) {
        PyErr_SetString(PyExc_ValueError, "Must have > 0 sites");
        goto out;
    }
    /* ancestor_age */
    ancestor_age_array = (PyArrayObject *) PyArray_FROM_OTF(ancestor_age,
            NPY_UINT32, NPY_ARRAY_IN_ARRAY);
    if (ancestor_age_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(ancestor_age_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(ancestor_age_array);
    num_ancestors = shape[0];
    if (num_ancestors < 1) {
        PyErr_SetString(PyExc_ValueError, "Must have > 0 ancestors");
        goto out;
    }
    /* focal_site */
    focal_site_array = (PyArrayObject *) PyArray_FROM_OTF(focal_site, NPY_UINT32,
            NPY_ARRAY_IN_ARRAY);
    if (focal_site_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(focal_site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(focal_site_array);
    num_focal_sites = shape[0];
    /* focal_site_ancestor */
    focal_site_ancestor_array = (PyArrayObject *) PyArray_FROM_OTF(focal_site_ancestor,
            NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (focal_site_ancestor_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(focal_site_ancestor_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(focal_site_ancestor_array);
    if (shape[0] != num_focal_sites) {
        PyErr_SetString(PyExc_ValueError, "Incorrect number of focal sites.");
        goto out;
    }
    /* site */
    site_array = (PyArrayObject *) PyArray_FROM_OTF(site, NPY_UINT32,
            NPY_ARRAY_IN_ARRAY);
    if (site_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(site_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(site_array);
    total_segments = shape[0];
    /* start */
    start_array = (PyArrayObject *) PyArray_FROM_OTF(start, NPY_INT32,
            NPY_ARRAY_IN_ARRAY);
    if (start_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(start_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(start_array);
    if (shape[0] != total_segments) {
        PyErr_SetString(PyExc_ValueError, "input start wrong size");
        goto out;
    }
    /* end */
    end_array = (PyArrayObject *) PyArray_FROM_OTF(end, NPY_INT32,
            NPY_ARRAY_IN_ARRAY);
    if (end_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(end_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(end_array);
    if (shape[0] != total_segments) {
        PyErr_SetString(PyExc_ValueError, "input end wrong size");
        goto out;
    }

    self->store = PyMem_Malloc(sizeof(ancestor_store_t));
    if (self->store == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = ancestor_store_alloc(self->store,
        num_sites, (double *) PyArray_DATA(position_array),
        num_ancestors,
        (uint32_t *) PyArray_DATA(ancestor_age_array),
        num_focal_sites,
        (int32_t *) PyArray_DATA(focal_site_ancestor_array),
        (uint32_t *) PyArray_DATA(focal_site_array),
        total_segments,
        (uint32_t *) PyArray_DATA(site_array),
        (int32_t *) PyArray_DATA(start_array),
        (int32_t *) PyArray_DATA(end_array));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    Py_XDECREF(position_array);
    Py_XDECREF(focal_site_array);
    Py_XDECREF(ancestor_age_array);
    Py_XDECREF(site_array);
    Py_XDECREF(start_array);
    Py_XDECREF(end_array);
    return ret;
}

static PyObject *
AncestorStore_get_ancestor(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"id", "haplotype", NULL};
    PyObject *py_focal_sites = NULL;
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    unsigned long ancestor_id;
    site_id_t start, end, *focal_sites;
    size_t num_sites, num_older_ancestors, num_focal_sites;
    npy_intp *shape;

    if (AncestorStore_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kO!", kwlist,
            &ancestor_id, &PyArray_Type, &haplotype)) {
        goto fail;
    }
    num_sites = self->store->num_sites;
    haplotype_array = (PyArrayObject *) PyArray_FROM_OTF(haplotype, NPY_INT8,
            NPY_ARRAY_INOUT_ARRAY);
    if (haplotype_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(haplotype_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(haplotype_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError, "input haplotype wrong size");
        goto fail;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_store_get_ancestor(self->store, ancestor_id,
        (int8_t *) PyArray_DATA(haplotype_array), &start, &end,
        &num_older_ancestors, &num_focal_sites, &focal_sites);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    py_focal_sites = convert_site_id_list(focal_sites, num_focal_sites);
    if (py_focal_sites == NULL) {
        goto fail;
    }
    Py_DECREF(haplotype_array);
    return Py_BuildValue("iikO", (int) start, (int) end,
            (unsigned long) num_older_ancestors, py_focal_sites);
fail:
    PyArray_XDECREF_ERR(haplotype_array);
    return NULL;
}

static PyObject *
AncestorStore_get_epoch_ancestors(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"epoch", "ancestors", NULL};
    PyObject *ancestors = NULL;
    PyArrayObject *ancestors_array = NULL;
    int epoch;
    size_t num_ancestors;
    npy_intp *shape;

    if (AncestorStore_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO!", kwlist,
            &epoch, &PyArray_Type, &ancestors)) {
        goto fail;
    }
    ancestors_array = (PyArrayObject *) PyArray_FROM_OTF(ancestors, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (ancestors_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(ancestors_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(ancestors_array);
    if (shape[0] < self->store->num_ancestors) {
        PyErr_SetString(PyExc_ValueError, "input ancestors potentially too small");
        goto fail;
    }
    if (epoch < 0 || epoch > self->store->num_epochs) {
        PyErr_SetString(PyExc_ValueError, "epoch out of bounds");
        goto fail;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_store_get_epoch_ancestors(self->store, epoch,
        (ancestor_id_t *) PyArray_DATA(ancestors_array), &num_ancestors);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(ancestors_array);
    return Py_BuildValue("k", (unsigned long) num_ancestors);
fail:
    PyArray_XDECREF_ERR(ancestors_array);
    return NULL;
}


static PyObject *
AncestorStore_get_age(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = {"ancestor_id", NULL};
    unsigned long ancestor_id;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &ancestor_id)) {
        goto out;
    }
    if (ancestor_id >= self->store->num_ancestors) {
        PyErr_SetString(PyExc_ValueError, "ancestor id out of bounds.");
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store->ancestors.age[ancestor_id]);
out:
    return ret;
}

static PyObject *
AncestorStore_get_state(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = {"site_id", "ancestor_id", NULL};
    unsigned long ancestor_id;
    unsigned long site_id;
    allele_t state;
    int err;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kk", kwlist, &site_id, &ancestor_id)) {
        goto out;
    }
    if (ancestor_id >= self->store->num_ancestors) {
        PyErr_SetString(PyExc_ValueError, "ancestor id out of bounds.");
        goto out;
    }
    if (site_id >= self->store->num_sites) {
        PyErr_SetString(PyExc_ValueError, "site id out of bounds.");
        goto out;
    }
    err = ancestor_store_get_state(self->store, site_id, ancestor_id, &state);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("i", (int) state);
out:
    return ret;
}


static PyObject *
AncestorStore_get_site(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    PyObject *ret = NULL;
    static char *kwlist[] = {"id", NULL};
    unsigned long site_id;
    site_state_t *site;
    size_t num_sites;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &site_id)) {
        goto out;
    }
    num_sites = self->store->num_sites;
    if (site_id >= num_sites) {
        PyErr_SetString(PyExc_ValueError, "site id out of bounds.");
        goto out;
    }
    site = self->store->sites + site_id;
    ret = convert_site(site);
out:
    return ret;
}

static PyObject *
AncestorStore_get_num_sites(AncestorStore *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store->num_sites);
out:
    return ret;
}

static PyObject *
AncestorStore_get_num_ancestors(AncestorStore *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store->num_ancestors);
out:
    return ret;
}

static PyObject *
AncestorStore_get_num_epochs(AncestorStore *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store->num_epochs);
out:
    return ret;
}

static PyObject *
AncestorStore_get_total_segments(AncestorStore *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store->total_segments);
out:
    return ret;
}

static PyObject *
AncestorStore_get_max_site_segments(AncestorStore *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store->max_site_segments);
out:
    return ret;
}

static PyObject *
AncestorStore_get_mean_site_segments(AncestorStore *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("d", self->store->mean_site_segments);
out:
    return ret;
}



static PyObject *
AncestorStore_get_total_memory(AncestorStore *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue("k", (unsigned long) self->store->total_memory);
out:
    return ret;
}

static PyMemberDef AncestorStore_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef AncestorStore_getsetters[] = {
    {"num_sites", (getter) AncestorStore_get_num_sites, NULL, "The number of sites."},
    {"num_ancestors", (getter) AncestorStore_get_num_ancestors, NULL, "The number of ancestors."},
    {"num_epochs", (getter) AncestorStore_get_num_epochs, NULL, "The number of epochs."},
    {"total_segments", (getter) AncestorStore_get_total_segments, NULL,
        "The total number of segments across all sites."},
    {"max_site_segments", (getter) AncestorStore_get_max_site_segments, NULL,
        "The maximum number of segments for a site."},
    {"mean_site_segments", (getter) AncestorStore_get_mean_site_segments, NULL,
        "The mean number of segments per site."},
    {"total_memory", (getter) AncestorStore_get_total_memory, NULL,
        "The total amount of memory used by this store."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorStore_methods[] = {
    {"get_ancestor", (PyCFunction) AncestorStore_get_ancestor,
        METH_VARARGS|METH_KEYWORDS,
        "Decodes the specified ancestor into the numpy array."},
    {"get_epoch_ancestors", (PyCFunction) AncestorStore_get_epoch_ancestors,
        METH_VARARGS|METH_KEYWORDS,
        "Returns the number of ancestors in the specified epoch and fills the "
        "specified array with the ancestors in question."},
    {"get_age", (PyCFunction) AncestorStore_get_age,
        METH_VARARGS|METH_KEYWORDS,
        "Returns the age of the specified ancestor."},
    {"get_state", (PyCFunction) AncestorStore_get_state,
        METH_VARARGS|METH_KEYWORDS,
        "Returns state of the specified ancestor and the specified locus."},
    {"get_site", (PyCFunction) AncestorStore_get_site,
        METH_VARARGS|METH_KEYWORDS,
        "Returns the encoded states for the specified site."},
    {NULL}  /* Sentinel */
};

static PyTypeObject AncestorStoreType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.AncestorStore",             /* tp_name */
    sizeof(AncestorStore),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)AncestorStore_dealloc, /* tp_dealloc */
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
    "AncestorStore objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    AncestorStore_methods,             /* tp_methods */
    AncestorStore_members,             /* tp_members */
    AncestorStore_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)AncestorStore_init,      /* tp_init */
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
    static char *kwlist[] = {"num_sites", "max_nodes", "max_edges", NULL};

    self->tree_sequence_builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kkk", kwlist,
                &num_sites, &max_nodes, &max_edges)) {
        goto out;
    }
    self->tree_sequence_builder = PyMem_Malloc(sizeof(tree_sequence_builder_t));
    if (self->tree_sequence_builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = tree_sequence_builder_alloc(self->tree_sequence_builder, num_sites,
            max_nodes, max_edges);
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
        /* site mutations */
        "site", "node", NULL};
    unsigned long num_nodes;
    double time;
    size_t j, num_edges, num_site_mutations;
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
    edge_t *edges = NULL;
    site_mutation_t *site_mutations = NULL;
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
    num_site_mutations = shape[0];

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
    if (shape[0] != num_site_mutations) {
        PyErr_SetString(PyExc_ValueError, "node wrong size");
        goto out;
    }

    /* Build the structs for the input */
    edges = PyMem_Malloc(num_edges * sizeof(edge_t));
    for (j = 0; j < num_edges; j++) {
        edges[j].left = ((site_id_t *) PyArray_DATA(left_array))[j];
        edges[j].right = ((site_id_t *) PyArray_DATA(right_array))[j];
        edges[j].parent = ((node_id_t *) PyArray_DATA(parent_array))[j];
        edges[j].child = ((node_id_t *) PyArray_DATA(child_array))[j];
    }

    /* Build the structs for the input */
    site_mutations = PyMem_Malloc(num_site_mutations * sizeof(site_mutation_t));
    for (j = 0; j < num_site_mutations; j++) {
        site_mutations[j].site = ((site_id_t *) PyArray_DATA(site_array))[j];
        site_mutations[j].node = ((node_id_t *) PyArray_DATA(node_array))[j];
    }
    Py_BEGIN_ALLOW_THREADS
    err = tree_sequence_builder_update(self->tree_sequence_builder,
            num_nodes, time, num_edges, edges, num_site_mutations,
            site_mutations);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    if (edges != NULL) {
        PyMem_Free(edges);
    }
    if (site_mutations != NULL) {
        PyMem_Free(site_mutations);
    }
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
    static char *kwlist[] = {"node", "haplotype", NULL};
    int node;
    PyObject *haplotype = NULL;
    PyArrayObject *haplotype_array = NULL;
    npy_intp *shape;
    size_t j, num_output_edges;
    edge_t *output_edges;
    PyArrayObject *left = NULL;
    PyArrayObject *right = NULL;
    PyArrayObject *parent = NULL;
    PyArrayObject *child = NULL;
    npy_intp dims[1];

    if (AncestorMatcher_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iO", kwlist, &node, &haplotype)) {
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
            (allele_t *) PyArray_DATA(haplotype_array), node,
            &num_output_edges, &output_edges);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    dims[0] = num_output_edges;
    left = (PyArrayObject *) PyArray_EMPTY(1, dims, NPY_UINT32, 0);
    right = (PyArrayObject *) PyArray_EMPTY(1, dims, NPY_UINT32, 0);
    parent = (PyArrayObject *) PyArray_EMPTY(1, dims, NPY_INT32, 0);
    child = (PyArrayObject *) PyArray_EMPTY(1, dims, NPY_INT32, 0);
    if (left == NULL || right == NULL || parent == NULL || child == NULL) {
        goto out;
    }
    for (j = 0; j < num_output_edges; j++) {
        ((site_id_t *) PyArray_DATA(left))[j] = output_edges[j].left;
        ((site_id_t *) PyArray_DATA(right))[j] = output_edges[j].right;
        ((node_id_t *) PyArray_DATA(parent))[j] = output_edges[j].parent;
        ((node_id_t *) PyArray_DATA(child))[j] = output_edges[j].child;
    }

    ret = Py_BuildValue("OOOO", left, right, parent, child);
out:
    Py_XDECREF(haplotype_array);
    Py_XDECREF(left);
    Py_XDECREF(right);
    Py_XDECREF(parent);
    Py_XDECREF(child);
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

static PyMemberDef AncestorMatcher_members[] = {
    {NULL}  /* Sentinel */

};

static PyGetSetDef AncestorMatcher_getsetters[] = {
    {"mean_traceback_size", (getter) AncestorMatcher_get_mean_traceback_size,
        NULL, "The mean size of the traceback per site."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorMatcher_methods[] = {
    {"find_path", (PyCFunction) AncestorMatcher_find_path,
        METH_VARARGS|METH_KEYWORDS,
        "Returns a best match path for the specified haplotype through the ancestors."},
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
 * ReferencePanel
 *===================================================================
 */

static int
ReferencePanel_check_state(ReferencePanel *self)
{
    int ret = 0;
    if (self->reference_panel == NULL) {
        PyErr_SetString(PyExc_SystemError, "ReferencePanel not initialised");
        ret = -1;
    }
    return ret;
}

static void
ReferencePanel_dealloc(ReferencePanel* self)
{
    if (self->reference_panel != NULL) {
        reference_panel_free(self->reference_panel);
        PyMem_Free(self->reference_panel);
        self->reference_panel = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
ReferencePanel_init(ReferencePanel *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"array", "positions", "sequence_length", NULL};
    PyObject *haplotypes_input =  NULL;
    PyArrayObject *haplotypes_array = NULL;
    PyObject *positions_input =  NULL;
    PyArrayObject *positions_array = NULL;
    npy_intp *shape;
    double sequence_length;
    uint32_t num_samples, num_sites;

    self->reference_panel = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOd", kwlist,
                &haplotypes_input, &positions_input, &sequence_length)) {
        goto out;
    }
    haplotypes_array = (PyArrayObject *) PyArray_FROM_OTF(haplotypes_input,
            NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    if (haplotypes_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(haplotypes_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Dim != 2");
        goto out;
    }
    shape = PyArray_DIMS(haplotypes_array);
    num_sites = (uint32_t) shape[1];
    num_samples = (uint32_t) shape[0];
    self->num_samples = (unsigned int) num_samples;
    self->num_sites = (unsigned int) num_sites;
    self->sequence_length = sequence_length;
    if (num_samples < 1) {
        PyErr_Format(PyExc_ValueError, "At least one haplotype required.");
        goto out;
    }
    if (num_sites < 1) {
        PyErr_Format(PyExc_ValueError, "At least one site required.");
        goto out;
    }
    positions_array = (PyArrayObject *) PyArray_FROM_OTF(positions_input,
            NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (positions_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(positions_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(positions_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError, "Wrong dimensions for positions");
        goto out;
    }
    self->reference_panel = PyMem_Malloc(sizeof(reference_panel_t));
    if (self->reference_panel == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = reference_panel_alloc(self->reference_panel, num_samples, num_sites,
           PyArray_DATA(haplotypes_array), PyArray_DATA(positions_array));
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    self->num_haplotypes = self->reference_panel->num_haplotypes;
    /* reference_panel_print_state(self->reference_panel); */
    ret = 0;
out:
    Py_XDECREF(haplotypes_array);
    Py_XDECREF(positions_array);
    return ret;
}

static PyObject *
ReferencePanel_get_haplotypes(ReferencePanel *self)
{
    PyObject *ret = NULL;
    PyArrayObject *array = NULL;
    npy_intp dims[2];

    if (ReferencePanel_check_state(self) != 0) {
        goto out;
    }
    dims[0] = self->num_haplotypes;
    dims[1] = self->num_sites;
    /* TODO we could avoid copying the memory here by using PyArray_SimpleNewFromData
     * and incrementing the refcount on this object. See
     * https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#c.PyArray_SimpleNewFromData
     */
    array = (PyArrayObject *) PyArray_SimpleNew(2, dims, NPY_UINT8);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->reference_panel->haplotypes,
            self->num_sites * self->num_haplotypes * sizeof(uint8_t));
    ret = (PyObject*) array;
out:
    return ret;
}

static PyObject *
ReferencePanel_get_positions(ReferencePanel *self)
{
    PyObject *ret = NULL;
    PyArrayObject *array = NULL;
    npy_intp dims[1];

    if (ReferencePanel_check_state(self) != 0) {
        goto out;
    }
    dims[0] = self->num_sites + 2;
    /* TODO we could avoid copying the memory here by using PyArray_SimpleNewFromData
     * and incrementing the refcount on this object. See
     * https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html#c.PyArray_SimpleNewFromData
     */
    array = (PyArrayObject *) PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->reference_panel->positions,
            (self->num_sites + 2) * sizeof(double));
    ret = (PyObject*) array;
out:
    return ret;
}

static PyMemberDef ReferencePanel_members[] = {
    {"num_haplotypes", T_UINT, offsetof(ReferencePanel, num_haplotypes), 0,
         "Number of haplotypes in the reference panel."},
    {"num_sites", T_UINT, offsetof(ReferencePanel, num_sites), 0,
         "Number of sites in the reference panel."},
    {"num_samples", T_UINT, offsetof(ReferencePanel, num_samples), 0,
         "Number of samples in the reference panel."},
    {"sequence_length", T_DOUBLE, offsetof(ReferencePanel, sequence_length), 0,
         "Length of the sequence."},
    {NULL}  /* Sentinel */
};

static PyMethodDef ReferencePanel_methods[] = {
    {"get_haplotypes", (PyCFunction) ReferencePanel_get_haplotypes, METH_NOARGS,
        "Returns a numpy array of the haplotypes in the panel."},
    {"get_positions", (PyCFunction) ReferencePanel_get_positions, METH_NOARGS,
        "Returns a numpy array of the positions in the panel."},
    {NULL}  /* Sentinel */
};

static PyTypeObject ReferencePanelType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.ReferencePanel",             /* tp_name */
    sizeof(ReferencePanel),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)ReferencePanel_dealloc, /* tp_dealloc */
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
    "ReferencePanel objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    ReferencePanel_methods,             /* tp_methods */
    ReferencePanel_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)ReferencePanel_init,      /* tp_init */
};

/*===================================================================
 * Threader
 *===================================================================
 */

static int
Threader_check_state(Threader *self)
{
    int ret = 0;
    if (self->threader == NULL) {
        PyErr_SetString(PyExc_SystemError, "Threader not initialised");
        ret = -1;
        goto out;
    }
    ret = ReferencePanel_check_state(self->reference_panel);
out:
    return ret;
}

static void
Threader_dealloc(Threader* self)
{
    if (self->threader != NULL) {
        threader_free(self->threader);
        PyMem_Free(self->threader);
        self->threader = NULL;
    }
    Py_XDECREF(self->reference_panel);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
Threader_init(Threader *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"reference_panel", NULL};
    ReferencePanel *reference_panel = NULL;

    self->threader = NULL;
    self->reference_panel = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist,
                &ReferencePanelType, &reference_panel)) {
        goto out;
    }
    self->reference_panel = reference_panel;
    Py_INCREF(self->reference_panel);
    if (ReferencePanel_check_state(self->reference_panel) != 0) {
        goto out;
    }
    self->threader = PyMem_Malloc(sizeof(threader_t));
    if (self->threader == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = threader_alloc(self->threader, self->reference_panel->reference_panel);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
Threader_run(Threader *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"haplotype_index", "panel_size", "recombination_rate",
        "error_probablilty", "path", "algorithm", NULL};
    PyObject *path = NULL;
    PyArrayObject *path_array = NULL;
    unsigned int panel_size, haplotype_index;
    double recombination_rate;
    uint32_t *mutations = NULL;
    uint32_t num_mutations;
    double error_probablilty;
    PyObject *mutations_array = NULL;
    npy_intp *shape;
    npy_intp mutations_shape;
    int algorithm = 0;

    if (Threader_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "IIddO|i", kwlist,
            &haplotype_index, &panel_size, &recombination_rate, &error_probablilty,
            &path, &algorithm)) {
        goto out;
    }
    if (haplotype_index >= self->threader->reference_panel->num_haplotypes) {
        PyErr_SetString(PyExc_ValueError, "haplotype_index out of bounds");
        goto out;
    }
    path_array = (PyArrayObject *) PyArray_FROM_OTF(path, NPY_UINT32,
            NPY_ARRAY_OUT_ARRAY);
    if (path_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(path_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(path_array);
    if (((uint32_t) shape[0]) != self->threader->reference_panel->num_sites) {
        PyErr_SetString(PyExc_ValueError, "input path wrong size");
        goto out;
    }
    mutations = PyMem_Malloc(self->threader->reference_panel->num_sites * sizeof(uint32_t));
    if (mutations == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = threader_run(self->threader, (uint32_t) haplotype_index,
            (uint32_t) panel_size, recombination_rate, error_probablilty,
            (uint32_t *) PyArray_DATA(path_array), &num_mutations, mutations);
    Py_END_ALLOW_THREADS
    err = 0;
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    mutations_shape = (npy_intp) num_mutations;
    mutations_array = PyArray_EMPTY(1, &mutations_shape, NPY_UINT32, 0);
    if (mutations_array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA((PyArrayObject *) mutations_array), mutations,
            num_mutations * sizeof(uint32_t));
    ret = mutations_array;
out:
    Py_XDECREF(path_array);
    return ret;
}

static PyObject *
Threader_get_traceback(Threader *self, void *closure)
{
    PyObject *ret = NULL;
    PyArrayObject *array;
    npy_intp dims[2];
    size_t N, m;

    if (Threader_check_state(self) != 0) {
        goto out;
    }
    N = self->threader->reference_panel->num_haplotypes;
    m = self->threader->reference_panel->num_sites;
    dims[0] = N;
    dims[1] = m;
    array = (PyArrayObject *) PyArray_EMPTY(2, dims, NPY_UINT32, 0);
    if (array == NULL) {
        goto out;
    }
    memcpy(PyArray_DATA(array), self->threader->T, N * m * sizeof(uint32_t));
    ret = (PyObject *) array;
out:
    return ret;
}

static PyGetSetDef Threader_getsetters[] = {
    {"traceback", (getter) Threader_get_traceback, NULL, "The flags array"},
    {NULL}  /* Sentinel */
};

static PyMemberDef Threader_members[] = {
    {NULL}  /* Sentinel */
};

static PyMethodDef Threader_methods[] = {
    {"run", (PyCFunction) Threader_run, METH_VARARGS|METH_KEYWORDS,
        "Threads a given haplotype through the first k haplotypes."},
    {NULL}  /* Sentinel */
};

static PyTypeObject ThreaderType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_tsinfer.Threader",             /* tp_name */
    sizeof(Threader),             /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Threader_dealloc, /* tp_dealloc */
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
    "Threader objects",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    Threader_methods,             /* tp_methods */
    Threader_members,             /* tp_members */
    Threader_getsetters,          /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Threader_init,      /* tp_init */
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyObject *
sort_ancestors(PyObject *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"ancestors", "permutation", NULL};
    PyObject *ancestors = NULL;
    PyObject *permutation = NULL;
    PyArrayObject *permutation_array = NULL;
    PyArrayObject *ancestors_array = NULL;
    size_t num_ancestors, num_sites;
    npy_intp *shape;
    ancestor_sorter_t sorter;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OO!", kwlist,
            &ancestors, &PyArray_Type, &permutation)) {
        goto fail;
    }
    ancestors_array = (PyArrayObject *) PyArray_FROM_OTF(ancestors, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
    if (ancestors_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(ancestors_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Dim != 2");
        goto fail;
    }
    shape = PyArray_DIMS(ancestors_array);
    num_ancestors = shape[0];
    num_sites = shape[1];
    if (num_ancestors < 1) {
        PyErr_SetString(PyExc_ValueError, "num_ancestors < 1");
        goto fail;
    }
    if (num_sites < 1) {
        PyErr_SetString(PyExc_ValueError, "num_sites < 1");
        goto fail;
    }
    permutation_array = (PyArrayObject *) PyArray_FROM_OTF(permutation, NPY_UINT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (permutation_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(permutation_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(permutation_array);
    if (shape[0] != num_ancestors) {
        PyErr_SetString(PyExc_ValueError, "input permutation wrong size");
        goto fail;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_sorter_alloc(&sorter, num_ancestors, num_sites,
        (int8_t *) PyArray_DATA(ancestors_array),
        (uint32_t *) PyArray_DATA(permutation_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        ancestor_sorter_free(&sorter);
        goto fail;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_sorter_sort(&sorter);
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        ancestor_sorter_free(&sorter);
        goto fail;
    }
    ancestor_sorter_free(&sorter);
    Py_DECREF(ancestors_array);
    Py_DECREF(permutation_array);
    return Py_BuildValue("");
fail:
    Py_XDECREF(ancestors_array);
    PyArray_XDECREF_ERR(permutation_array);
    return NULL;

}

static PyMethodDef tsinfer_methods[] = {
    {"sort_ancestors", (PyCFunction) sort_ancestors,
        METH_VARARGS|METH_KEYWORDS,
        "Sorts the ancestors in the array to minimise block breaks and "
        "return the resulting permutation" },
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
    /* AncestorStoreBuilder type */
    AncestorStoreBuilderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AncestorStoreBuilderType) < 0) {
        INITERROR;
    }
    Py_INCREF(&AncestorStoreBuilderType);
    PyModule_AddObject(module, "AncestorStoreBuilder", (PyObject *) &AncestorStoreBuilderType);
    /* AncestorStore type */
    AncestorStoreType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&AncestorStoreType) < 0) {
        INITERROR;
    }
    Py_INCREF(&AncestorStoreType);
    PyModule_AddObject(module, "AncestorStore", (PyObject *) &AncestorStoreType);
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
    /* ReferencePanel type */
    ReferencePanelType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ReferencePanelType) < 0) {
        INITERROR;
    }
    Py_INCREF(&ReferencePanelType);
    PyModule_AddObject(module, "ReferencePanel", (PyObject *) &ReferencePanelType);
    /* Threader type */
    ThreaderType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ThreaderType) < 0) {
        INITERROR;
    }
    Py_INCREF(&ThreaderType);
    PyModule_AddObject(module, "Threader", (PyObject *) &ThreaderType);
    TsinfLibraryError = PyErr_NewException("_tsinfer.LibraryError", NULL, NULL);
    Py_INCREF(TsinfLibraryError);
    PyModule_AddObject(module, "LibraryError", TsinfLibraryError);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
