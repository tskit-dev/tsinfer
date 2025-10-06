/*
 * This implementation is largely adapted from tskit's tsk_blkalloc
 * (MIT License). It is a near-copy with minor adjustments for use in
 * tsinfer (e.g., tracking total_size and naming). See
 * lib/subprojects/tskit for the original sources and licensing.
 */
#include "tsinfer.h"
#include <stdlib.h>
#include <stdio.h>

void
tsi_blkalloc_print_state(tsi_blkalloc_t *self, FILE *out)
{
    fprintf(out, "Block allocator%p::\n", (void *) self);
    fprintf(out, "\ttop = %lld\n", (long long) self->top);
    fprintf(out, "\tchunk_size = %lld\n", (long long) self->chunk_size);
    fprintf(out, "\tnum_chunks = %lld\n", (long long) self->num_chunks);
    fprintf(out, "\ttotal_allocated = %lld\n", (long long) self->total_allocated);
    fprintf(out, "\ttotal_size = %lld\n", (long long) self->total_size);
}

int WARN_UNUSED
tsi_blkalloc_reset(tsi_blkalloc_t *self)
{
    int ret = 0;

    self->top = 0;
    self->current_chunk = 0;
    self->total_allocated = 0;
    return ret;
}

int WARN_UNUSED
tsi_blkalloc_init(tsi_blkalloc_t *self, size_t chunk_size)
{
    int ret = 0;

    tsk_memset(self, 0, sizeof(tsi_blkalloc_t));
    if (chunk_size < 1) {
        ret = TSK_ERR_BAD_PARAM_VALUE;
        goto out;
    }
    self->chunk_size = chunk_size;
    self->top = 0;
    self->current_chunk = 0;
    self->total_allocated = 0;
    self->total_size = 0;
    self->num_chunks = 0;
    self->mem_chunks = malloc(sizeof(char *));
    if (self->mem_chunks == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    self->mem_chunks[0] = malloc(chunk_size);
    if (self->mem_chunks[0] == NULL) {
        ret = TSK_ERR_NO_MEMORY;
        goto out;
    }
    self->num_chunks = 1;
    self->total_size = chunk_size + sizeof(void *);
out:
    return ret;
}

void *WARN_UNUSED
tsi_blkalloc_get(tsi_blkalloc_t *self, size_t size)
{
    void *ret = NULL;
    void *p;

    if (size > self->chunk_size) {
        goto out;
    }
    if ((self->top + size) > self->chunk_size) {
        if (self->current_chunk == (self->num_chunks - 1)) {
            p = realloc(self->mem_chunks, (self->num_chunks + 1) * sizeof(void *));
            if (p == NULL) {
                goto out;
            }
            self->mem_chunks = p;
            p = malloc(self->chunk_size);
            if (p == NULL) {
                goto out;
            }
            self->mem_chunks[self->num_chunks] = p;
            self->num_chunks++;
            self->total_size += self->chunk_size + sizeof(void *);
        }
        self->current_chunk++;
        self->top = 0;
    }
    ret = self->mem_chunks[self->current_chunk] + self->top;
    self->top += size;
    self->total_allocated += size;
out:
    return ret;
}

void
tsi_blkalloc_free(tsi_blkalloc_t *self)
{
    size_t j;

    for (j = 0; j < self->num_chunks; j++) {
        if (self->mem_chunks[j] != NULL) {
            free(self->mem_chunks[j]);
        }
    }
    if (self->mem_chunks != NULL) {
        free(self->mem_chunks);
    }
}
