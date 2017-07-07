#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static void
segment_list_check_state(segment_list_t *self)
{
    segment_list_node_t *u;
    size_t j;

    j = 0;
    for (u = self->head; u != NULL; u = u->next) {
        assert(u->start < u->end);
        if (u->next != NULL) {
            assert(u->end < u->next->start);
        }
        j++;
    }
    assert(j == self->length);
}

int
segment_list_print_state(segment_list_t *self, FILE *out)
{
    segment_list_node_t *u;

    fprintf(out, "Segment list\n");
    fprintf(out, "length = %d\n", (int) self->length);
    object_heap_print_state(&self->heap, out);
    for (u = self->head; u != NULL; u = u->next) {
        fprintf(out, "(%d, %d)", u->start, u->end);
    }
    fprintf(out, "\n");
    segment_list_check_state(self);
    return 0;
}

int
segment_list_alloc(segment_list_t *self, size_t block_size)
{
    int ret = 0;

    memset(self, 0, sizeof(segment_list_t));
    self->block_size = block_size;
    ret = object_heap_init(&self->heap, sizeof(segment_list_node_t),
           self->block_size, NULL);
    if (ret != 0) {
        goto out;
    }
out:
    return ret;
}

int
segment_list_free(segment_list_t *self)
{
    object_heap_free(&self->heap);
    return 0;
}

static inline segment_list_node_t * WARN_UNUSED
segment_list_alloc_segment(segment_list_t *self, site_id_t start, site_id_t end)
{
    segment_list_node_t *ret = NULL;

    if (object_heap_empty(&self->heap)) {
        if (object_heap_expand(&self->heap) != 0) {
            goto out;
        }
    }
    ret = (segment_list_node_t *) object_heap_alloc_object(&self->heap);
    if (ret == NULL) {
        goto out;
    }
    ret->start = start;
    ret->end = end;
    ret->next = NULL;
    self->length++;
out:
    return ret;
}

static inline void
segment_list_free_segment(segment_list_t *self, segment_list_node_t *seg)
{
    object_heap_free_object(&self->heap, seg);
}

int
segment_list_append(segment_list_t *self, site_id_t start, site_id_t end)
{
    int ret = 0;
    segment_list_node_t *u;

    if (self->head == NULL) {
        self->head = segment_list_alloc_segment(self, start, end);
        if (self->head == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        self->tail = self->head;
    } else {
        assert(self->tail->end <= start);
        if (self->tail->end == start) {
            self->tail->end = end;
        } else {
            u = segment_list_alloc_segment(self, start, end);
            if (u == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            self->tail->next = u;
            self->tail = u;
        }
    }
out:
    return ret;
}

int
segment_list_clear(segment_list_t *self)
{
    int ret = 0;
    segment_list_node_t *v = self->head;
    segment_list_node_t *tmp;

    while (v != NULL) {
        tmp = v;
        v = v->next;
        segment_list_free_segment(self, tmp);
    }
    self->head = NULL;
    self->tail = NULL;
    self->length = 0;
    return ret;
}
