#include "tsinfer.h"
#include "err.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static void
print_segment_chain(segment_t *head, bool as_int, FILE *out)
{
    segment_t *u = head;

    while (u != NULL) {
        if (as_int) {
            fprintf(out, "(%d-%d:%d)", u->start, u->end, (int) u->value);
        } else {
            fprintf(out, "(%d-%d:%f)", u->start, u->end, u->value);
        }
        u = u->next;
        if (u != NULL) {
            fprintf(out, "=>");
        }
    }
}

static void
ancestor_store_builder_check_state(ancestor_store_builder_t *self)
{
    site_id_t l;
    segment_t *u;
    size_t total_segments = 0;

    for (l = 0; l < self->num_sites; l++) {
        u = self->sites_head[l];
        assert(u != NULL);
        assert(u->start == 0);
        while (u->next != NULL) {
            assert(u->next->start >= u->end);
            assert(u->value != -1);
            u = u->next;
            total_segments++;
        }
        total_segments++;
        assert(u->value != -1);
        assert(self->sites_tail[l] == u);
    }
    assert(total_segments == object_heap_get_num_allocated(&self->segment_heap));
    assert(total_segments == self->total_segments);
}

int
ancestor_store_builder_print_state(ancestor_store_builder_t *self, FILE *out)
{
    site_id_t l;
    segment_t *seg;
    int num_segments;

    fprintf(out, "Ancestor store builder state\n");
    fprintf(out, "num_sites = %d\n", (int) self->num_sites);
    fprintf(out, "num_ancestors = %d\n", (int) self->num_ancestors);
    fprintf(out, "total_segments = %d\n", (int) self->total_segments);
    fprintf(out, "Segment heap:\n");
    object_heap_print_state(&self->segment_heap, out);
    fprintf(out, "Sites:\n");
    for (l = 0; l < self->num_sites; l++) {
        num_segments = 0;
        seg = self->sites_head[l];
        while (seg != NULL) {
            seg = seg->next;
            num_segments++;
        }
        fprintf(out, "%d\t(%d):", l, num_segments);
        print_segment_chain(self->sites_head[l], true, out);
        fprintf(out, "\n");
    }

    ancestor_store_builder_check_state(self);
    return 0;
}

static inline segment_t * WARN_UNUSED
ancestor_store_builder_alloc_segment(ancestor_store_builder_t *self, ancestor_id_t start,
        ancestor_id_t end, double value)
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
    ret->value = value;
    ret->next = NULL;
out:
    return ret;
}

int
ancestor_store_builder_alloc(ancestor_store_builder_t *self, size_t num_sites,
        size_t segment_block_size)
{
    int ret = 0;
    site_id_t l;

    memset(self, 0, sizeof(ancestor_store_builder_t));
    self->num_sites = num_sites;
    self->num_ancestors = 1;
    self->sites_head = malloc(self->num_sites * sizeof(segment_t *));
    self->sites_tail = malloc(self->num_sites * sizeof(segment_t *));
    if (self->sites_head == NULL || self->sites_tail == NULL) {
        ret = TSI_ERR_NO_MEMORY;
        goto out;

    }
    ret = object_heap_init(&self->segment_heap, sizeof(segment_t),
           segment_block_size, NULL);
    if (ret != 0) {
        goto out;
    }
    for (l = 0; l < self->num_sites; l++) {
        self->sites_head[l] = ancestor_store_builder_alloc_segment(self, 0, 1, 0);
        if (self->sites_head[l] == NULL) {
            ret = TSI_ERR_NO_MEMORY;
            goto out;
        }
        self->sites_tail[l] = self->sites_head[l];
    }
    self->total_segments = self->num_sites;
out:
    return ret;
}

int
ancestor_store_builder_free(ancestor_store_builder_t *self)
{
    tsi_safe_free(self->sites_head);
    tsi_safe_free(self->sites_tail);
    object_heap_free(&self->segment_heap);

    return 0;
}

int
ancestor_store_builder_add(ancestor_store_builder_t *self, allele_t *ancestor)
{
    int ret = 0;
    site_id_t l;
    segment_t *tail;
    ancestor_id_t n = (ancestor_id_t) self->num_ancestors;

    for (l = 0; l < self->num_sites; l++) {
        if (ancestor[l] != -1) {
            tail = self->sites_tail[l];
            if (tail->end == n && tail->value == (double) ancestor[l]) {
                tail->end++;
            } else {
                tail = ancestor_store_builder_alloc_segment(self, n, n + 1, ancestor[l]);
                self->sites_tail[l]->next = tail;
                self->sites_tail[l] = tail;
                self->total_segments++;
            }
        }
    }
    self->num_ancestors++;
    return ret;
}

int
ancestor_store_builder_dump(ancestor_store_builder_t *self, site_id_t *site, ancestor_id_t *start,
        ancestor_id_t *end, allele_t *state)
{
    int ret = 0;
    site_id_t l;
    segment_t *u;
    size_t k;

    k = 0;
    for (l = 0; l < self->num_sites; l++) {
        u = self->sites_head[l];
        while (u != NULL) {
            assert(k < self->total_segments);
            site[k] = l;
            start[k] = u->start;
            end[k] = u->end;
            state[k] = (allele_t) u->value;
            u = u->next;
            k++;
        }
    }
    return ret;
}
