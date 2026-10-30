
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

#define MODULE_DOC "Low-level tsinfer interface."

static PyObject *TsinfLibraryError;

typedef struct {
    PyObject_HEAD
    ancestor_builder_t *builder;
} AncestorBuilder;

static void
handle_library_error(int err)
{
    if (err == TSI_ERR_NO_MEMORY) {
        PyErr_NoMemory();
    } else {
        PyErr_Format(TsinfLibraryError, "%s", tsi_strerror(err));
    }
}

static FILE *
make_file(PyObject *fileobj, const char *mode)
{
    FILE *ret = NULL;
    FILE *file = NULL;
    int fileobj_fd, new_fd;

    fileobj_fd = PyObject_AsFileDescriptor(fileobj);
    if (fileobj_fd == -1) {
        goto out;
    }
    new_fd = dup(fileobj_fd);
    if (new_fd == -1) {
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    file = fdopen(new_fd, mode);
    if (file == NULL) {
        (void) close(new_fd);
        PyErr_SetFromErrno(PyExc_OSError);
        goto out;
    }
    ret = file;
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
AncestorBuilder_dealloc(AncestorBuilder *self)
{
    if (self->builder != NULL) {
        ancestor_builder_free(self->builder);
        PyMem_Free(self->builder);
        self->builder = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *) self);
}

static int
AncestorBuilder_init(AncestorBuilder *self, PyObject *args, PyObject *kwds)
{
    int ret = -1;
    int err;
    static char *kwlist[]
        = { "num_samples", "max_sites", "genotype_encoding", "mmap_fd", NULL };
    int num_samples, max_sites;
    int genotype_encoding = 0;
    int flags = 0;
    int mmap_fd = -1;

    self->builder = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii|ii", kwlist, &num_samples,
            &max_sites, &genotype_encoding, &mmap_fd)) {
        goto out;
    }
    self->builder = PyMem_Malloc(sizeof(ancestor_builder_t));
    if (self->builder == NULL) {
        PyErr_NoMemory();
        goto out;
    }
    flags = genotype_encoding;
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_alloc(self->builder, num_samples, max_sites, mmap_fd, flags);
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
    static char *kwlist[] = { "time", "genotypes", NULL };
    PyObject *ret = NULL;
    double time;
    PyObject *genotypes = NULL;
    PyArrayObject *genotypes_array = NULL;
    npy_intp *shape;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "dO", kwlist, &time, &genotypes)) {
        goto out;
    }
    genotypes_array
        = (PyArrayObject *) PyArray_FROM_OTF(genotypes, NPY_INT8, NPY_ARRAY_IN_ARRAY);
    if (genotypes_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(genotypes_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(genotypes_array);
    if (shape[0] != (npy_intp) self->builder->num_samples) {
        PyErr_SetString(PyExc_ValueError, "genotypes array wrong size.");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_add_site(
        self->builder, time, (allele_t *) PyArray_DATA(genotypes_array));
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
    PyObject *ret = NULL;
    static char *kwlist[] = { "focal_sites", "ancestor", NULL };
    PyObject *ancestor = NULL;
    PyArrayObject *ancestor_array = NULL;
    PyObject *focal_sites = NULL;
    PyArrayObject *focal_sites_array = NULL;
    size_t num_focal_sites;
    size_t num_sites;
    tsk_id_t start, end;
    npy_intp *shape;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OO!", kwlist, &focal_sites, &PyArray_Type, &ancestor)) {
        goto out;
    }
    num_sites = self->builder->num_sites;
    focal_sites_array
        = (PyArrayObject *) PyArray_FROM_OTF(focal_sites, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    if (focal_sites_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(focal_sites_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(focal_sites_array);
    num_focal_sites = shape[0];
    if (num_focal_sites == 0 || num_focal_sites > num_sites) {
        PyErr_SetString(PyExc_ValueError, "num_focal_sites must > 0 and <= num_sites");
        goto out;
    }
    ancestor_array
        = (PyArrayObject *) PyArray_FROM_OTF(ancestor, NPY_INT8, NPY_ARRAY_INOUT_ARRAY);
    if (ancestor_array == NULL) {
        goto out;
    }
    if (PyArray_NDIM(ancestor_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dim != 1");
        goto out;
    }
    shape = PyArray_DIMS(ancestor_array);
    if (shape[0] != (npy_intp) num_sites) {
        PyErr_SetString(PyExc_ValueError, "input ancestor wrong size");
        goto out;
    }
    Py_BEGIN_ALLOW_THREADS
    err = ancestor_builder_make_ancestor(self->builder, num_focal_sites,
        (int32_t *) PyArray_DATA(focal_sites_array), &start, &end,
        (int8_t *) PyArray_DATA(ancestor_array));
    Py_END_ALLOW_THREADS
    if (err != 0) {
        handle_library_error(err);
        goto out;
    }
    ret = Py_BuildValue("ii", start, end);
out:
    Py_XDECREF(focal_sites_array);
    Py_XDECREF(ancestor_array);
    return ret;
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

static PyObject *
AncestorBuilder_get_memsize(AncestorBuilder *self, void *closure)
{
    PyObject *ret = NULL;

    if (AncestorBuilder_check_state(self) != 0) {
        goto out;
    }
    ret = Py_BuildValue(
        "k", (unsigned long) ancestor_builder_get_memsize(self->builder));
out:
    return ret;
}

static PyMemberDef AncestorBuilder_members[] = {
    { NULL } /* Sentinel */
};

static PyGetSetDef AncestorBuilder_getsetters[] = {
    { "num_sites", (getter) AncestorBuilder_get_num_sites, NULL,
        "The number of sites." },
    { "num_ancestors", (getter) AncestorBuilder_get_num_ancestors, NULL,
        "The number of ancestors." },
    { "mem_size", (getter) AncestorBuilder_get_memsize, NULL,
        "The number of allocated bytes." },
    { NULL } /* Sentinel */
};

static PyMethodDef AncestorBuilder_methods[] = {
    { "add_site", (PyCFunction) AncestorBuilder_add_site, METH_VARARGS | METH_KEYWORDS,
        "Adds the specified site to this ancestor builder." },
    { "make_ancestor", (PyCFunction) AncestorBuilder_make_ancestor,
        METH_VARARGS | METH_KEYWORDS, "Makes the specified ancestor." },
    { "ancestor_descriptors", (PyCFunction) AncestorBuilder_ancestor_descriptors,
        METH_NOARGS, "Returns a list of ancestor (frequency, focal_sites) tuples." },
    { NULL } /* Sentinel */
};

static PyTypeObject AncestorBuilderType = {
    PyVarObject_HEAD_INIT(NULL, 0) "_tsinfer.AncestorBuilder", /* tp_name */
    sizeof(AncestorBuilder),                                   /* tp_basicsize */
    0,                                                         /* tp_itemsize */
    (destructor) AncestorBuilder_dealloc,                      /* tp_dealloc */
    0,                                                         /* tp_print */
    0,                                                         /* tp_getattr */
    0,                                                         /* tp_setattr */
    0,                                                         /* tp_reserved */
    0,                                                         /* tp_repr */
    0,                                                         /* tp_as_number */
    0,                                                         /* tp_as_sequence */
    0,                                                         /* tp_as_mapping */
    0,                                                         /* tp_hash  */
    0,                                                         /* tp_call */
    0,                                                         /* tp_str */
    0,                                                         /* tp_getattro */
    0,                                                         /* tp_setattro */
    0,                                                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,                                        /* tp_flags */
    "AncestorBuilder objects",                                 /* tp_doc */
    0,                                                         /* tp_traverse */
    0,                                                         /* tp_clear */
    0,                                                         /* tp_richcompare */
    0,                                                         /* tp_weaklistoffset */
    0,                                                         /* tp_iter */
    0,                                                         /* tp_iternext */
    AncestorBuilder_methods,                                   /* tp_methods */
    AncestorBuilder_members,                                   /* tp_members */
    AncestorBuilder_getsetters,                                /* tp_getset */
    0,                                                         /* tp_base */
    0,                                                         /* tp_dict */
    0,                                                         /* tp_descr_get */
    0,                                                         /* tp_descr_set */
    0,                                                         /* tp_dictoffset */
    (initproc) AncestorBuilder_init,                           /* tp_init */
};

/*===================================================================
 * Module level code.
 *===================================================================
 */

static PyMethodDef tsinfer_methods[] = {
    { NULL } /* Sentinel */
};

/* Initialisation code supports Python 2.x and 3.x. The framework uses the
 * recommended structure from http://docs.python.org/howto/cporting.html.
 * I've ignored the point about storing state in globals, as the examples
 * from the Python documentation still use this idiom.
 */

#if PY_MAJOR_VERSION >= 3

static struct PyModuleDef tsinfermodule
    = { PyModuleDef_HEAD_INIT, "_tsinfer", /* name of module */
          MODULE_DOC,                      /* module documentation, may be NULL */
          -1, tsinfer_methods, NULL, NULL, NULL, NULL };

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

    TsinfLibraryError = PyErr_NewException("_tsinfer.LibraryError", NULL, NULL);
    Py_INCREF(TsinfLibraryError);
    PyModule_AddObject(module, "LibraryError", TsinfLibraryError);

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
