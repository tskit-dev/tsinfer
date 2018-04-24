/*
** Copyright (C) 2018 University of Oxford
**
** This file is part of tsinfer.
**
** tsinfer is free software: you can redistribute it and/or modify
** it under the terms of the GNU General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
**
** tsinfer is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU General Public License for more details.
**
** You should have received a copy of the GNU General Public License
** along with tsinfer.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#include "err.h"
#include "block_allocator.h"

void
block_allocator_print_state(block_allocator_t *self, FILE *out)
{
    fprintf(out, "Block allocator%p::\n", (void *) self);
    fprintf(out, "\ttop = %d\n", (int) self->top);
    fprintf(out, "\tchunk_size = %d\n", (int) self->chunk_size);
    fprintf(out, "\tnum_chunks = %d\n", (int) self->num_chunks);
    fprintf(out, "\ttotal_allocated = %d\n", (int) self->total_allocated);
    fprintf(out, "\ttotal_size = %d\n", (int) self->total_size);
}

int WARN_UNUSED
block_allocator_reset(block_allocator_t *self)
{
    int ret = 0;

    self->top = 0;
    self->current_chunk = 0;
    self->total_allocated = 0;
    return ret;
}

int WARN_UNUSED
block_allocator_alloc(block_allocator_t *self, size_t chunk_size)
{
    int ret = 0;

    assert(chunk_size > 0);
    memset(self, 0, sizeof(block_allocator_t));
    self->chunk_size = chunk_size;
    self->top = 0;
    self->current_chunk = 0;
    self->total_allocated = 0;
    self->total_size = 0;
    self->num_chunks = 0;
    self->mem_chunks = malloc(sizeof(char *));
    if (self->mem_chunks == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->mem_chunks[0] = malloc(chunk_size);
    if (self->mem_chunks[0] == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
    self->num_chunks = 1;
    self->total_size = chunk_size + sizeof(void *);
out:
    return ret;
}

void * WARN_UNUSED
block_allocator_get(block_allocator_t *self, size_t size)
{
    void *ret = NULL;
    void *p;

    assert(size < self->chunk_size);
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
block_allocator_free(block_allocator_t *self)
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
