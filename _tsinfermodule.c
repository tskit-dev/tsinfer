
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
    ancestor_store_t *store;
} AncestorStore;

typedef struct {
    PyObject_HEAD
    ancestor_matcher_t *matcher;
    AncestorStore *store;
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
    static char *kwlist[] = {"num_sites", NULL};
    unsigned long num_sites;

    self->store = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "k", kwlist, &num_sites)) {
        goto out;
    }
    if (num_sites < 1) {
        PyErr_SetString(PyExc_ValueError, "Must have > 0 sites");
        goto out;
    }
    self->store = PyMem_Malloc(sizeof(ancestor_store_t));
    if (self->store == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = ancestor_store_alloc(self->store, num_sites);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
AncestorStore_init_build(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    int err;
    PyObject *ret = NULL;
    static char *kwlist[] = {"segment_block_size", NULL};
    unsigned long segment_block_size = 100;

    if (AncestorStore_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|k", kwlist, &segment_block_size)) {
        goto out;
    }
    err = ancestor_store_init_build(self->store, segment_block_size);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("");
out:
    return ret;
}

static PyObject *
AncestorStore_add(AncestorStore *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"ancestor", NULL};
    PyObject *ancestor = NULL;
    PyArrayObject *ancestor_array = NULL;
    npy_intp *shape;

    if (AncestorStore_check_state(self) != 0) {
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
    if (shape[0] != self->store->num_sites) {
        PyErr_SetString(PyExc_ValueError, "input ancestor wrong size");
        goto fail;
    }
    err = ancestor_store_add(self->store, (int8_t *) PyArray_DATA(ancestor_array));
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

static PyMemberDef AncestorStore_members[] = {
    {NULL}  /* Sentinel */
};

static PyGetSetDef AncestorStore_getsetters[] = {
    {"num_sites", (getter) AncestorStore_get_num_sites, NULL, "The number of sites."},
    {"num_ancestors", (getter) AncestorStore_get_num_ancestors, NULL, "The number of ancestors."},
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorStore_methods[] = {
    {"init_build", (PyCFunction) AncestorStore_init_build,
        METH_VARARGS|METH_KEYWORDS,
        "Initialises the build process and adds the ultimate ancestor."},
    {"add", (PyCFunction) AncestorStore_add,
        METH_VARARGS|METH_KEYWORDS,
        "Adds the specified ancestor."},
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
 * AncestorMatcher
 *===================================================================
 */

static int
AncestorMatcher_check_state(AncestorMatcher *self)
{
    int ret = 0;
    if (self->matcher == NULL) {
        PyErr_SetString(PyExc_SystemError, "AncestorMatcher not initialised");
        ret = -1;
    }
    return ret;
}

static void
AncestorMatcher_dealloc(AncestorMatcher* self)
{
    if (self->matcher != NULL) {
        ancestor_matcher_free(self->matcher);
        PyMem_Free(self->matcher);
        self->matcher = NULL;
    }
    Py_XDECREF(self->store);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int
AncestorMatcher_init(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[] = {"store", "recombination_rate", "mutation_rate", NULL};
    AncestorStore *store = NULL;
    double recombination_rate;
    double mutation_rate;

    self->matcher = NULL;
    self->store = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!dd", kwlist,
                &AncestorStoreType, &store, &recombination_rate, &mutation_rate)) {
        goto out;
    }
    self->store = store;
    Py_INCREF(self->store);
    if (AncestorStore_check_state(self->store) != 0) {
        goto out;
    }
    self->matcher = PyMem_Malloc(sizeof(ancestor_matcher_t));
    if (self->matcher == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    err = ancestor_matcher_alloc(self->matcher, self->store->store, recombination_rate,
            mutation_rate);
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = 0;
out:
    return ret;
}

static PyObject *
AncestorMatcher_best_path(AncestorMatcher *self, PyObject *args, PyObject *kwds)
{
    int err;
    static char *kwlist[] = {"num_ancestors", "haplotype", "path", "mutation_sites", NULL};
    PyObject *haplotype = NULL;
    PyObject *path = NULL;
    PyObject *mutation_sites = NULL;
    PyArrayObject *path_array = NULL;
    PyArrayObject *haplotype_array = NULL;
    PyArrayObject *mutation_sites_array = NULL;
    unsigned long num_ancestors;
    size_t num_mutations, num_sites;
    npy_intp *shape;

    if (AncestorMatcher_check_state(self) != 0) {
        goto fail;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "kOO!O!", kwlist,
            &num_ancestors, &haplotype, &PyArray_Type, &path, &PyArray_Type,
            &mutation_sites)) {
        goto fail;
    }
    num_sites = self->matcher->store->num_sites;
    haplotype_array = (PyArrayObject *) PyArray_FROM_OTF(haplotype, NPY_INT8,
            NPY_ARRAY_IN_ARRAY);
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
    path_array = (PyArrayObject *) PyArray_FROM_OTF(path, NPY_INT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (path_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(path_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(path_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError, "input path wrong size");
        goto fail;
    }
    mutation_sites_array = (PyArrayObject *) PyArray_FROM_OTF(mutation_sites, NPY_UINT32,
            NPY_ARRAY_INOUT_ARRAY);
    if (mutation_sites_array == NULL) {
        goto fail;
    }
    if (PyArray_NDIM(mutation_sites_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto fail;
    }
    shape = PyArray_DIMS(mutation_sites_array);
    if (shape[0] != num_sites) {
        PyErr_SetString(PyExc_ValueError, "input mutation_sites wrong size");
        goto fail;
    }
    err = ancestor_matcher_best_path(self->matcher, num_ancestors,
        (int8_t *) PyArray_DATA(haplotype_array),
        (int32_t *) PyArray_DATA(path_array),
        &num_mutations, (uint32_t *) PyArray_DATA(mutation_sites_array));
    if (err != 0) {
        handle_library_error(err);
        goto fail;
    }
    Py_DECREF(haplotype_array);
    Py_DECREF(path_array);
    Py_DECREF(mutation_sites_array);
    return Py_BuildValue("k", (unsigned long) num_mutations);
fail:
    Py_XDECREF(haplotype_array);
    PyArray_XDECREF_ERR(path_array);
    PyArray_XDECREF_ERR(mutation_sites_array);
    return NULL;
}

static PyMemberDef AncestorMatcher_members[] = {
    {NULL}  /* Sentinel */
};

static PyMethodDef AncestorMatcher_methods[] = {
    {"best_path", (PyCFunction) AncestorMatcher_best_path,
        METH_VARARGS|METH_KEYWORDS,
        "Fills the specified numpy array with the best path through the ancestors."},
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
    0,                         /* tp_getset */
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
