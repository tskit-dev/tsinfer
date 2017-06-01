
#ifndef OBJECT_HEAP_H
#define OBJECT_HEAP_H

#include <stdio.h>
#include <string.h>
#include <assert.h>

typedef struct {
    size_t object_size;
    size_t block_size; /* number of objects in a block */
    size_t top;
    size_t size;
    size_t num_blocks;
    void **heap;
    char **mem_blocks;
    void (*init_object)(void **obj, size_t index);
} object_heap_t;

extern size_t object_heap_get_num_allocated(object_heap_t *self);
extern void object_heap_print_state(object_heap_t *self, FILE *out);
extern int object_heap_expand(object_heap_t *self);
extern void * object_heap_get_object(object_heap_t *self, size_t index);
extern int object_heap_empty(object_heap_t *self);
extern void * object_heap_alloc_object(object_heap_t *self);
extern void object_heap_free_object(object_heap_t *self, void *obj);
extern int object_heap_init(object_heap_t *self, size_t object_size, size_t block_size,
        void (*init_object)(void **, size_t));
extern void object_heap_free(object_heap_t *self);

#endif
