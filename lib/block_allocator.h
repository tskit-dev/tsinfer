#ifndef BLOCK_ALLOCATOR_H
#define BLOCK_ALLOCATOR_H

/* This is a simple allocator that is optimised to efficiently allocate a
 * large number of small objects without large numbers of calls to malloc.
 * The allocator mallocs memory in chunks of a configurable size. When
 * responding to calls to get(), it will return a chunk of this memory.
 * This memory cannot be subsequently handed back to the allocator. However,
 * all memory allocated by the allocator can be returned at once by calling
 * reset.
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>

typedef struct {
    size_t chunk_size;        /* number of bytes per chunk */
    size_t top;               /* the offset of the next available byte in the current chunk */
    size_t current_chunk;     /* the index of the chunk currently being used */
    size_t total_size;        /* the total number of bytes allocated + overhead. */
    size_t total_allocated;   /* the total number of bytes allocated. */
    size_t num_chunks;        /* the number of memory chunks. */
    char **mem_chunks;        /* the memory chunks */
} block_allocator_t;

extern void block_allocator_print_state(block_allocator_t *self, FILE *out);
extern int block_allocator_reset(block_allocator_t *self);
extern int block_allocator_alloc(block_allocator_t *self, size_t chunk_size);
extern void * block_allocator_get(block_allocator_t *self, size_t size);
extern void block_allocator_free(block_allocator_t *self);

#endif
