#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>


int
traceback_print_state(traceback_t *self, FILE *out)
{
    size_t l;
    segment_t *v;

    fprintf(out, "Traceback\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    object_heap_print_state(&self->segment_heap, out);
    for (l = 0; l < self->num_sites; l++) {
        v = self->sites_head[l];
        if (v != NULL) {
            fprintf(out, "%d\t%d\t", (int) l, self->best_match[l]);
            while (v != NULL) {
                fprintf(out, "(%d,%d)", (int) v->start, (int) v->end);
                v = v->next;
            }
            fprintf(out, "\n");
        }
    }
    return 0;
}

int
traceback_alloc(traceback_t *self, size_t num_sites, size_t segment_block_size)
{
    int ret = 0;

    memset(self, 0, sizeof(traceback_t));
    self->num_sites = num_sites;
    self->segment_block_size = segment_block_size;
    ret = object_heap_init(&self->segment_heap, sizeof(segment_t),
           self->segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    self->best_match = malloc(self->num_sites * sizeof(ancestor_id_t));
    self->sites_head = calloc(self->num_sites, sizeof(segment_t *));
    self->sites_tail = calloc(self->num_sites, sizeof(segment_t *));
    if (self->best_match == NULL
            || self->sites_head == NULL || self->sites_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;
    }
out:
    return ret;
}

int
traceback_free(traceback_t *self)
{
    object_heap_free(&self->segment_heap);
    tsi_safe_free(self->best_match);
    tsi_safe_free(self->sites_head);
    tsi_safe_free(self->sites_tail);
    return 0;
}

static inline segment_t * WARN_UNUSED
traceback_alloc_segment(traceback_t *self, ancestor_id_t start, ancestor_id_t end)
{
    segment_t *ret = NULL;

    if (object_heap_empty(&self->segment_heap)) {
        if (object_heap_expand(&self->segment_heap) != 0) {
            goto out;
        }
    }
    ret = (segment_t *) object_heap_alloc_object(&self->segment_heap);
    if (ret == NULL) {
        goto out;
    }
    ret->start = start;
    ret->end = end;
    ret->next = NULL;
out:
    return ret;
}

static inline void
traceback_free_segment(traceback_t *self, segment_t *seg)
{
    object_heap_free_object(&self->segment_heap, seg);
}

int
traceback_reset(traceback_t *self)
{
    size_t l;
    segment_t *v, *tmp;

    for (l = 0; l < self->num_sites; l++) {
        v = self->sites_head[l];
        while (v != NULL) {
            tmp = v;
            v = v->next;
            traceback_free_segment(self, tmp);
        }
        self->sites_head[l] = NULL;
        self->sites_tail[l] = NULL;
    }
    return 0;
}

int
traceback_set_best_match(traceback_t *self, site_id_t site_id, ancestor_id_t best_match)
{
    self->best_match[site_id] = best_match;
    return 0;
}

int
traceback_add_recombination(traceback_t *self, site_id_t site_id, ancestor_id_t start,
        ancestor_id_t end)
{
    int ret = 0;
    segment_t **head = self->sites_head;
    segment_t **tail = self->sites_tail;
    segment_t *tmp;
    assert(head != NULL);
    assert(tail != NULL);
    assert(site_id < self->num_sites);

    if (head[site_id] == NULL) {
        head[site_id] = traceback_alloc_segment(self, start, end);
        if (head[site_id] == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        tail[site_id] = head[site_id];
    } else {
        assert(tail[site_id] != NULL);
        if (tail[site_id]->end == start) {
            tail[site_id]->end = end;
        } else {
            tmp = traceback_alloc_segment(self, start, end);
            if (tmp == NULL) {
                ret = TSI_ERR_NO_MEMORY;
                goto out;
            }
            tail[site_id]->next = tmp;
            tail[site_id] = tmp;
        }
    }
out:
    return ret;
}
